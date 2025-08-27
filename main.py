import os
import itertools
import glob
import time

import numpy as np
import torch
import pywt
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from scipy import stats
from sklearn.utils import shuffle
from tqdm import tqdm

from baetorch.baetorch.models_v2.bae_ensemble import BAE_Ensemble
from baetorch.baetorch.models_v2.bae_mcdropout import BAE_MCDropout
from baetorch.baetorch.models_v2.bae_sghmc import BAE_SGHMC
from baetorch.baetorch.models_v2.bae_vi import BAE_VI
from baetorch.baetorch.models_v2.vae import VAE
from baetorch.baetorch.models_v2.outlier_proba import BAE_Outlier_Proba
from baetorch.baetorch.util.convert_dataloader import convert_dataloader
from baetorch.baetorch.util.seed import bae_set_seed
from baetorch.baetorch.util.misc import time_method


# ==============================================================================
# 全局参数配置 (Global Parameter Configuration)
# ==============================================================================

# --- 随机种子 ---
BAE_SEED = 100
bae_set_seed(BAE_SEED)

# --- 文件与路径设置 ---
DATA_PATH = "dataset/CCD/"
RESULT_DIR = "result/test"
SAVED_MODELS_DIR = "saved_models/"
SENSOR_NAME = "CCD"

# --- 数据集与图像参数 ---
IMAGE_TARGET_SIZE = (160, 160)
TIME_SERIES_GROUP_SIZE = 10  # 构建时序信号的图像分组大小
MAX_IMAGES_NORMAL = 1100
MAX_IMAGES_BALLING = 350
MAX_IMAGES_HIGH_DILUTION = 350

# --- 模型超参数 ---
GRID_PARAMS = {
    "random_seed": [BAE_SEED],
    "latent_factor": [0.5],
    "bae_type": ["mcd"],  
    "full_likelihood": ["mse"],  
}
NUM_EPOCHS = 100
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-11
USE_CUDA = torch.cuda.is_available()

# --- 自适应阈值参数 ---
ADAPTIVE_T = 30          # Z-检验的窗口大小 (原始图像)
ADAPTIVE_M = 100         # 定期更新阈值的周期 (原始图像)
ADAPTIVE_T_WAVELET = 3   # Z-检验的窗口大小 (小波系数)
ADAPTIVE_M_WAVELET = 10  # 定期更新阈值的周期 (小波系数)
UNCERTAINTY_THRESHOLD = 0.001  # 用于过滤样本的不确定性阈值

# ==============================================================================
# 辅助函数与映射关系 (Helper Functions & Mappings)
# ==============================================================================

def create_dir_if_not_exists(directory):
    """如果目录不存在，则创建它。"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def print_memory_usage(stage="Unknown"):
    """打印当前 GPU 内存使用情况。"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{stage} - 分配内存: {allocated:.3f} GiB, 预留内存: {reserved:.3f} GiB")

# --- BAE 模型参数映射 ---
FULL_LIKELIHOODS = ["mse", "homo-gauss", "hetero-gauss", "homo-tgauss", "hetero-tgauss", "bernoulli", "cbernoulli", "beta"]

HOMOSCEDESTIC_MODE_MAP = {
    "bernoulli": "none", "cbernoulli": "none", "homo-gauss": "every", "hetero-gauss": "none",
    "homo-tgauss": "none", "hetero-tgauss": "none", "mse": "none", "beta": "none"
}
LIKELIHOOD_MAP = {
    "bernoulli": "bernoulli", "cbernoulli": "cbernoulli", "homo-gauss": "gaussian",
    "hetero-gauss": "gaussian", "homo-tgauss": "truncated_gaussian",
    "hetero-tgauss": "truncated_gaussian", "mse": "gaussian", "beta": "beta"
}
TWIN_OUTPUT_MAP = {
    "bernoulli": False, "cbernoulli": False, "homo-gauss": False, "hetero-gauss": True,
    "homo-tgauss": False, "hetero-tgauss": True, "mse": False, "beta": True
}
BAE_TYPE_CLASSES = {
    "ens": BAE_Ensemble, "mcd": BAE_MCDropout, "sghmc": BAE_SGHMC,
    "vi": BAE_VI, "vae": VAE, "ae": BAE_Ensemble,
}
N_BAE_SAMPLES_MAP = {
    "ens": 5, "mcd": 100, "sghmc": 50,
    "vi": 100, "vae": 100, "ae": 1,
}

# ==============================================================================
# 数据加载与预处理 (Data Loading & Preprocessing)
# ==============================================================================

def load_images_from_directory(directory, target_size, max_images=None):
    """从指定目录加载、预处理并返回图像数据。"""
    images = []
    image_paths = sorted(glob.glob(os.path.join(directory, "*.tiff")))
    if max_images:
        image_paths = image_paths[:max_images]

    for img_path in image_paths:
        img = Image.open(img_path).convert("L")  # 转换为灰度图
        img = img.resize(target_size)
        img = np.array(img) / 255.0  # 归一化到 [0, 1]
        img = img[np.newaxis, :, :]  # 增加通道维度: (1, height, width)
        images.append(img)
    return np.array(images)

def extract_melt_pool_area(image):
    """从单张图像中提取熔池面积。"""
    img = image.squeeze(0) * 255
    img = img.astype(np.uint8)
    _, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    area = np.sum(binary > 0)
    return area

def build_time_series_from_images(images, group_size):
    """从图像序列构建熔池面积的时序信号。"""
    areas = [extract_melt_pool_area(img) for img in images]
    time_series = [
        areas[i:i+group_size]
        for i in range(0, len(areas), group_size)
        if len(areas[i:i+group_size]) == group_size
    ]
    return np.array(time_series)

def apply_wavelet_transform(time_series, wavelet='morl', scales=np.arange(1, 32)):
    """对时序信号应用小波变换。"""
    coefficients = [pywt.cwt(ts, scales, wavelet)[0] for ts in time_series]
    return np.array(coefficients)

# ==============================================================================
# 模型定义 (Model Definition)
# ==============================================================================

def define_bae_model(input_shape, latent_factor, bae_params):
    """根据输入形状和参数定义一个 BAE 模型。"""
    _, input_channels, height, width = input_shape

    # --- 动态计算卷积层输出和潜在维度 ---
    def calc_conv_output_dim(input_dim, kernel, stride, padding=0):
        return (input_dim - kernel + 2 * padding) // stride + 1

    # encoder 卷积层参数
    conv_params = {
        "conv_channels": [input_channels, 16, 32, 64],
        "conv_stride": [1, 2, 2],
        "conv_kernel": [3, 2, 2],
    }

    h_out, w_out = height, width
    for i in range(len(conv_params["conv_kernel"])):
        h_out = calc_conv_output_dim(h_out, conv_params["conv_kernel"][i], conv_params["conv_stride"][i])
        w_out = calc_conv_output_dim(w_out, conv_params["conv_kernel"][i], conv_params["conv_stride"][i])

    final_conv_channels = conv_params["conv_channels"][-1]
    latent_dim = int(final_conv_channels * h_out * w_out * latent_factor)

    # --- 定义模型架构 ---
    chain_params = [
        {
            "base": "conv2d",
            "input_dim": [height, width],
            "conv_channels": conv_params["conv_channels"],
            "conv_stride": conv_params["conv_stride"],
            "conv_kernel": conv_params["conv_kernel"],
            "activation": "leakyrelu",
            "norm": "none", "bias": False, "order": ["base", "norm", "activation"],
        },
        {
            "base": "linear",
            "architecture": [512, 256, latent_dim],
            "activation": "leakyrelu", "norm": "layer", "last_norm": "layer",
        },
    ]

    # --- 初始化 BAE 模型 ---
    bae_model = BAE_TYPE_CLASSES[bae_params["bae_type"]](
        chain_params=chain_params,
        last_activation="sigmoid",
        twin_output=bae_params["twin_output"],
        skip=bae_params.get("skip", False),
        use_cuda=USE_CUDA,
        homoscedestic_mode=bae_params["homoscedestic_mode"],
        likelihood=bae_params["likelihood"],
        weight_decay=WEIGHT_DECAY,
        num_samples=bae_params["n_bae_samples"],
        learning_rate=LEARNING_RATE,
        stochastic_seed=bae_params["random_seed"],
        anchored=bae_params.get("anchored", False),
    )
    return bae_model


# ======================================================================
# 自适应阈值
# ======================================================================

def compute_adaptive_threshold(data, labels, bae_model, bae_proba_model, T, M, result_dir, sensor_name, case_name, random_seed=42):
    """计算自适应阈值，并记录历史。"""
    # 固定初始阈值
    threshold = 0.5
    fixed_threshold = 0.5
    soft_threshold_upper = 0.9
    min_threshold = 0.5

    # 初始化历史记录
    threshold_history = [(0, threshold)]
    adaptive_accuracy_history = []
    fixed_accuracy_history = []

    # 窗口设置
    Upper_Bound = 1000
    z_w1 = []
    z_w2 = []
    t_w = []
    unstable_count = 0
    unstable_buffer = []

    # 打乱数据
    data, labels = shuffle(data, labels, random_state=random_seed)

    # 累积预测和真实标签
    cumulative_pred_adaptive = []
    cumulative_pred_fixed = []
    cumulative_true = []
    cumulative_unc = []

    # 逐个处理数据点
    for i, x in enumerate(data):
        x = x[np.newaxis, ...]
        score = bae_model.predict(x, select_keys=["nll"])
        score, unc = bae_proba_model.predict(score["nll"], norm_scaling=True)
        epi_scalar = float(np.asarray(unc["epi"]).mean())

        # 自适应阈值分类（初始判断）
        pred_adaptive = 1 if score > threshold else 0

        # 软阈值逻辑
        if score < threshold:  # 正常数据
            z_w2.append(float(score))
            t_w.append(float(score))
            unstable_count = 0  # 重置计数器
            unstable_buffer = []  # 清空缓冲区
            cumulative_pred_adaptive.append(pred_adaptive)
        elif threshold <= score <= soft_threshold_upper:  # 正常不稳定数据
            unstable_buffer.append((float(score), i))
            unstable_count += 1
            cumulative_pred_adaptive.append(pred_adaptive)  # 暂存初始标签
            if 3 >= unstable_count >= 1:  # 连续1~3个不稳定数据
                for score, idx in unstable_buffer[-1:]:
                    cumulative_pred_adaptive[idx] = 0
                    z_w2.append(score)
                    t_w.append(score)
                unstable_count = 0  # 重置计数器
                unstable_buffer = []  # 清空缓冲区
        else:  # 异常数据
            unstable_count = 0  # 重置计数器
            unstable_buffer = []  # 清空缓冲区
            cumulative_pred_adaptive.append(pred_adaptive)

        # 固定阈值分类
        pred_fixed = 1 if score > fixed_threshold else 0
        cumulative_pred_fixed.append(pred_fixed)

        # 真实标签
        cumulative_true.append(labels[i])

        # 不确定性
        cumulative_unc.append(epi_scalar)

        # t-test
        if len(z_w2) >= T:
            if len(z_w1) >= T:
                stat, p_value = stats.ttest_ind(z_w1, z_w2, equal_var=False)
                if p_value < 0.05:
                    threshold = np.mean(t_w) + 2 * np.std(t_w)
                    threshold = max(threshold, min_threshold)
                    if threshold>=1:
                        threshold=1
                    threshold_history.append((i + 1, threshold))
                    t_w = t_w[-T:]
                    z_w1 = z_w2
                    z_w2 = []
                else:
                    z_w2 = []
            else:
                z_w1.extend(z_w2)
                z_w2 = []

        # 定期更新
        if len(t_w) % M == 0 and len(t_w) > 0:
            threshold = np.mean(t_w) + 2 * np.std(t_w)
            threshold = max(threshold, min_threshold)
            if threshold>=1:
                threshold=1
            threshold_history.append((i + 1, threshold))

        # 窗口上限
        if len(t_w) > Upper_Bound:
            t_w.pop(0)

        # 计算精确率
        if len(cumulative_true) > 0:
            adaptive_accuracy = np.mean(np.array(cumulative_pred_adaptive) == np.array(cumulative_true))
            fixed_accuracy = np.mean(np.array(cumulative_pred_fixed) == np.array(cumulative_true))
            adaptive_accuracy_history.append(adaptive_accuracy)
            fixed_accuracy_history.append(fixed_accuracy)

    # 保存历史
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f"threshold_history_{sensor_name}_{case_name}.txt"), 'w', encoding="utf-8") as f:
        for idx, thresh in threshold_history:
            f.write(f"更新索引: {idx}, 阈值: {thresh:.4f}\n")
    with open(os.path.join(result_dir, f"accuracy_history_{sensor_name}_{case_name}.txt"), 'w', encoding="utf-8") as f:
        for i, (adaptive_acc, fixed_acc) in enumerate(zip(adaptive_accuracy_history, fixed_accuracy_history)):
            f.write(f"数据点索引: {i+1}, 自适应阈值准确率: {adaptive_acc:.4f}, 固定阈值准确率: {fixed_acc:.4f}\n")

    return cumulative_pred_adaptive, cumulative_pred_fixed, cumulative_true, threshold_history, adaptive_accuracy_history, fixed_accuracy_history, cumulative_unc


# ======================================================================
# 评估与打印辅助函数 (Fixed/Adaptive/Uncertainty)
# ======================================================================

def compute_fixed_threshold_labels(id_scores, ood_scores, threshold=0.5):
    """基于固定阈值将 ID=0, OOD=1，返回 (scores, labels, preds)。"""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))]).astype(int)
    preds = (scores >= threshold).astype(int)
    return scores, labels, preds

def binary_metrics(labels, preds):
    """二分类指标：精确率、召回率、准确率、F1。"""
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    accuracy = float((labels == preds).mean())
    f1 = f1_score(labels, preds, zero_division=0)
    return precision, recall, accuracy, f1

def print_metrics_table(rows, title="指标汇总"):
    """按表格格式打印指标。rows: [(name, acc, prec, rec, f1), ...]"""
    print(f"\n{title}：")
    print(f"{'模型':<28}{'准确率':>10}{'精确率':>12}{'召回率':>10}{'F1':>10}")
    print("-" * 90)
    for name, acc, prec, rec, f1 in rows:
        print(f"{name:<28}{acc:>10.3f}{prec:>12.3f}{rec:>10.3f}{f1:>10.3f}")

def build_proba_model(bae_model, x_ref):
    """基于参考 ID 数据拟合 outlier proba 模型。"""
    nll_key = "nll"
    ref_pred = bae_model.predict(x_ref, select_keys=[nll_key])[nll_key]
    proba_model = BAE_Outlier_Proba(dist_type="norm", norm_scaling=True, fit_per_bae_sample=True)
    proba_model.fit(ref_pred)
    return proba_model

def uncertainty_metrics(y_pred, y_true, unc_hist, unc_threshold):
    """不确定性过滤的指标计算与保留比例。"""
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    unc_scalar = np.asarray(unc_hist, dtype=float)  

    # 依据不确定性保留样本
    mask = unc_scalar <= unc_threshold

    yp, yt = y_pred[mask], y_true[mask]
    if len(yp) == 0 or len(np.unique(yt)) < 2:
        return None, None, None, None, 0.0

    p, r, a, f1 = binary_metrics(yt, yp)
    return p, r, a, f1, float(mask.mean())


# ==============================================================================
# 主执行逻辑 (Main Execution Logic)
# ==============================================================================

def main():
    """主函数，执行整个异常检测流程。"""
    # --- 创建结果目录 ---
    create_dir_if_not_exists(RESULT_DIR)
    create_dir_if_not_exists(SAVED_MODELS_DIR)

    # --- 清理 GPU 缓存 ---
    if USE_CUDA:
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print_memory_usage("初始化")

    # --- 网格搜索循环 ---
    for values in tqdm(itertools.product(*GRID_PARAMS.values()), desc="Grid Search"):
        exp_params = dict(zip(GRID_PARAMS.keys(), values))
        print("\n" + "="*50)
        print("当前实验参数:", exp_params)
        print("="*50)

        # --- 解包实验参数 ---
        random_seed = exp_params["random_seed"]
        latent_factor = exp_params["latent_factor"]
        bae_type = exp_params["bae_type"]
        full_likelihood_i = exp_params["full_likelihood"]

        bae_set_seed(random_seed)

        model_params = {
            "random_seed": random_seed,
            "bae_type": bae_type,
            "twin_output": TWIN_OUTPUT_MAP[full_likelihood_i],
            "homoscedestic_mode": HOMOSCEDESTIC_MODE_MAP[full_likelihood_i],
            "likelihood": LIKELIHOOD_MAP[full_likelihood_i],
            "n_bae_samples": N_BAE_SAMPLES_MAP[bae_type],
            "anchored": True if bae_type == "ens" else False
        }

        # ----------------------------------------------------------------------
        # 1. 数据准备 (Data Preparation)
        # ----------------------------------------------------------------------
        print("\n--- 1. 开始加载和预处理数据... ---")
        x_inlier_data = load_images_from_directory(
            os.path.join(DATA_PATH, "1normal-melting"), IMAGE_TARGET_SIZE, MAX_IMAGES_NORMAL
        )
        x_ood_balling_data = load_images_from_directory(
            os.path.join(DATA_PATH, "0balling-tendency"), IMAGE_TARGET_SIZE, MAX_IMAGES_BALLING
        )
        x_ood_high_dilution_data = load_images_from_directory(
            os.path.join(DATA_PATH, "3High-dilution-rate"), IMAGE_TARGET_SIZE, MAX_IMAGES_HIGH_DILUTION
        )

        # 划分训练集和测试集
        n_total = x_inlier_data.shape[0]
        split_index = int(n_total * 0.7)
        x_id_train_orig = x_inlier_data[:split_index]  # 70%
        x_id_test_orig = x_inlier_data[split_index:]   # 30%

        # 构建时序信号并进行小波变换
        train_time_series = build_time_series_from_images(x_id_train_orig, TIME_SERIES_GROUP_SIZE)
        test_time_series = build_time_series_from_images(x_id_test_orig, TIME_SERIES_GROUP_SIZE)
        ood_balling_time_series = build_time_series_from_images(x_ood_balling_data, TIME_SERIES_GROUP_SIZE)
        ood_high_dilution_time_series = build_time_series_from_images(x_ood_high_dilution_data, TIME_SERIES_GROUP_SIZE)

        train_wavelet_coef = apply_wavelet_transform(train_time_series)
        test_wavelet_coef = apply_wavelet_transform(test_time_series)
        ood_balling_wavelet_coef = apply_wavelet_transform(ood_balling_time_series)
        ood_high_dilution_wavelet_coef = apply_wavelet_transform(ood_high_dilution_time_series)

        # 小波系数归一化
        scaler = MinMaxScaler()
        train_wavelet_flat = train_wavelet_coef.reshape(train_wavelet_coef.shape[0], -1)
        train_wavelet_norm = scaler.fit_transform(train_wavelet_flat).reshape(train_wavelet_coef.shape)

        def transform_wavelet(coef, scaler):
            flat = coef.reshape(coef.shape[0], -1)
            norm = scaler.transform(flat).reshape(coef.shape)
            return norm[:, np.newaxis, :, :]  # 增加通道维度

        x_id_train_wav = transform_wavelet(train_wavelet_coef, scaler)
        x_id_test_wav = transform_wavelet(test_wavelet_coef, scaler)
        x_ood_balling_wav = transform_wavelet(ood_balling_wavelet_coef, scaler)
        x_ood_high_dilution_wav = transform_wavelet(ood_high_dilution_wavelet_coef, scaler)

        # 创建 DataLoader
        train_loader_orig = convert_dataloader(x_id_train_orig, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        train_loader_wav = convert_dataloader(x_id_train_wav, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        print("--- 数据准备完成 ---")

        # ----------------------------------------------------------------------
        # 2. 模型训练 (Model Training)
        # ----------------------------------------------------------------------
        print("\n--- 2. 开始模型定义与训练... ---")
        bae_model_original = define_bae_model(x_id_train_orig.shape, latent_factor, model_params)
        bae_model_wavelet = define_bae_model(x_id_train_wav.shape, latent_factor, model_params)

        print_memory_usage("模型初始化后")

        print("训练原始图像模型...")
        time_method(bae_model_original.fit,train_loader_orig, num_epochs=NUM_EPOCHS)
        bae_model_original.save_model_state(filename=f"original_{random_seed}_{SENSOR_NAME}.pt", folder_path=SAVED_MODELS_DIR)

        print("训练小波系数模型...")
        time_method(bae_model_wavelet.fit,train_loader_wav, num_epochs=NUM_EPOCHS)
        bae_model_wavelet.save_model_state(filename=f"wavelet_{random_seed}_{SENSOR_NAME}.pt", folder_path=SAVED_MODELS_DIR)

        # print("加载预训练模型...")
        # bae_model_original.load_model_state(filename=f"original_0bae_model_CCD_seed_520.pt", folder_path=SAVED_MODELS_DIR)
        # bae_model_wavelet.load_model_state(filename=f"wavelet_0bae_model_CCD_seed_520.pt", folder_path=SAVED_MODELS_DIR)

        print("--- 模型训练/加载完成 ---")

        # ----------------------------------------------------------------------
        # 3. 预测与评估 (Prediction & Evaluation)
        # ----------------------------------------------------------------------
        print("\n--- 3. 开始预测与评估... ---")

        def get_outlier_scores(model, x_ref, x_test, x_oods):
            """计算 ID 测试集/各 OOD 的异常概率与不确定性。"""
            nll_key = "nll"
            ref_pred = model.predict(x_ref, select_keys=[nll_key])
            test_pred = model.predict(x_test, select_keys=[nll_key])

            proba_model = BAE_Outlier_Proba(dist_type="norm", norm_scaling=True, fit_per_bae_sample=True)
            proba_model.fit(ref_pred[nll_key])

            id_proba_mean, id_proba_unc = proba_model.predict(test_pred[nll_key], norm_scaling=True)

            ood_probas = []
            for x_ood in x_oods:
                ood_pred = model.predict(x_ood, select_keys=[nll_key])
                ood_proba_mean, ood_proba_unc = proba_model.predict(ood_pred[nll_key], norm_scaling=True)
                ood_probas.append({"mean": ood_proba_mean, "unc": ood_proba_unc})

            return id_proba_mean, id_proba_unc, ood_probas

        # --- 计算异常分数 ---
        print("计算原始图像数据流的异常分数...")
        id_proba_mean_orig, id_proba_unc_orig, ood_probas_orig = get_outlier_scores(
            bae_model_original,
            x_id_train_orig,
            x_id_test_orig,
            [x_ood_balling_data, x_ood_high_dilution_data]
        )

        print("计算小波图像数据流的异常分数...")
        id_proba_mean_wav, id_proba_unc_wav, ood_probas_wav = get_outlier_scores(
            bae_model_wavelet,
            x_id_train_wav,
            x_id_test_wav,
            [x_ood_balling_wav, x_ood_high_dilution_wav]
        )

        print("--- 预测与评估完成 ---")

        # ----------------------------------------------------------------------
        # 4. 结果可视化与打印 (Results Visualization & Printing)
        # ----------------------------------------------------------------------
        print("\n--- 4. 开始生成结果与打印... ---")

        plt.rcParams.update({
            'font.size': 12, 'font.family': 'Times New Roman', 'axes.labelsize': 14,
            'axes.titlesize': 16, 'legend.fontsize': 10, 'xtick.labelsize': 12,
            'ytick.labelsize': 12, 'lines.linewidth': 2, 'axes.grid': False,
        })

        # === 固定阈值：四个任务的性能表 ===
        FIXED_THR = 0.5
        fixed_rows = []
        cases_fixed = [
            ("Balling (Original)",        id_proba_mean_orig, ood_probas_orig[0]["mean"]),
            ("High-Dilution (Original)",  id_proba_mean_orig, ood_probas_orig[1]["mean"]),
            ("Balling (Wavelet)",         id_proba_mean_wav,  ood_probas_wav[0]["mean"]),
            ("High-Dilution (Wavelet)",   id_proba_mean_wav,  ood_probas_wav[1]["mean"]),
        ]
        for name, id_scores, ood_scores in cases_fixed:
            _, labels, preds = compute_fixed_threshold_labels(id_scores, ood_scores, threshold=FIXED_THR)
            prec, rec, acc, f1 = binary_metrics(labels, preds)
            fixed_rows.append((name, acc, prec, rec, f1))
        print_metrics_table(fixed_rows, title=f"固定阈值指标")

        # === 自适应阈值：缓存首次运行的结果 ===
        # 1) 拼接模拟在线数据流
        sorted_idx_orig = np.argsort(id_proba_mean_orig)
        sorted_id_data_orig = x_id_test_orig[sorted_idx_orig]
        sorted_idx_wav = np.argsort(id_proba_mean_wav)
        sorted_id_data_wav = x_id_test_wav[sorted_idx_wav]

        # 2) per-case 的 proba 模型（用各自训练 ID 拟合）
        proba_orig = build_proba_model(bae_model_original, x_id_train_orig)
        proba_wav = build_proba_model(bae_model_wavelet, x_id_train_wav)

        adaptive_cases = [
            ("Balling Tendency (Original)",
             np.concatenate([sorted_id_data_orig, x_ood_balling_data], axis=0),
             np.concatenate([np.zeros(len(sorted_id_data_orig)), np.ones(len(x_ood_balling_data))]),
             bae_model_original, proba_orig, ADAPTIVE_T, ADAPTIVE_M),

            ("High Dilution Rate (Original)",
             np.concatenate([sorted_id_data_orig, x_ood_high_dilution_data], axis=0),
             np.concatenate([np.zeros(len(sorted_id_data_orig)), np.ones(len(x_ood_high_dilution_data))]),
             bae_model_original, proba_orig, ADAPTIVE_T, ADAPTIVE_M),

            ("Balling Tendency (Wavelet)",
             np.concatenate([sorted_id_data_wav, x_ood_balling_wav], axis=0),
             np.concatenate([np.zeros(len(sorted_id_data_wav)), np.ones(len(x_ood_balling_wav))]),
             bae_model_wavelet, proba_wav, ADAPTIVE_T_WAVELET, ADAPTIVE_M_WAVELET),

            ("High Dilution Rate (Wavelet)",
             np.concatenate([sorted_id_data_wav, x_ood_high_dilution_wav], axis=0),
             np.concatenate([np.zeros(len(sorted_id_data_wav)), np.ones(len(x_ood_high_dilution_wav))]),
             bae_model_wavelet, proba_wav, ADAPTIVE_T_WAVELET, ADAPTIVE_M_WAVELET),
        ]

        # 缓存：避免对相同 case 重复计算
        adaptive_cache = {}

        def get_adaptive_result(case_key, data_seq, labels_seq, mdl, pmodel, T, M):
            if case_key not in adaptive_cache:
                adaptive_cache[case_key] = compute_adaptive_threshold(
                    data=data_seq,
                    labels=labels_seq,
                    bae_model=mdl,
                    bae_proba_model=pmodel,
                    T=T, M=M,
                    result_dir=RESULT_DIR,
                    sensor_name=SENSOR_NAME,
                    case_name=case_key,
                    random_seed=BAE_SEED,
                )
            return adaptive_cache[case_key]

        # 打印自适应阈值指标与阈值历史预览
        adaptive_rows = []
        for (case_name, data_seq, labels_seq, mdl, pmodel, T, M) in adaptive_cases:
            pa, pf, yt, thr_hist, acc_a_hist, acc_f_hist, unc_hist = get_adaptive_result(
                case_name, data_seq, labels_seq, mdl, pmodel, T, M
            )

            # 自适应指标（使用 pa vs yt）
            prec, rec, acc, f1 = binary_metrics(np.array(yt), np.array(pa))
            adaptive_rows.append((f"{case_name} (Adaptive)", acc, prec, rec, f1))

        print_metrics_table(adaptive_rows, title="自适应阈值指标")

        # === 不确定性过滤 ===
        unc_rows = []
        for (case_name, data_seq, labels_seq, mdl, pmodel, T, M) in adaptive_cases:
            pa, pf, yt, thr_hist, acc_a_hist, acc_f_hist, unc_hist = adaptive_cache[case_name]
            res = uncertainty_metrics(pa, yt, unc_hist, UNCERTAINTY_THRESHOLD)
            if res[0] is None:
                print(f"\n[{case_name}] 不确定性过滤后类别不足（无法计算指标）。")
                continue
            p, r, a, f1, keep = res
            unc_rows.append((f"{case_name} (+Unc@{UNCERTAINTY_THRESHOLD})", a, p, r, f1))
            print(f"\n[{case_name}] 不确定性阈值={UNCERTAINTY_THRESHOLD:.4f} 的保留比例：{keep:.2%}")

        print_metrics_table(unc_rows, title="加入不确定性后的指标")

    print("\n所有实验已完成！")


if __name__ == "__main__":
    main()
