# UQAT-BAE based Anomaly Detection for Melt Pool Monitoring

## 📌 Overview

This repository implements an **uncertainty quantified and adaptive thresholding Bayesian autoencoder (UQAT-BAE)** framework for anomaly detection in **Directed Energy Deposition (DED)** melt pool monitoring. The approach integrates **wavelet-based time–frequency analysis** with adaptive thresholding and uncertainty filtering to achieve robust detection under process drift and varying defect types.

The workflow includes:

1. **Data loading & preprocessing** for both raw coaxial CCD images and their wavelet scalograms.
2. **Bayesian Autoencoder model training** on inlier (normal) melt pool data.
3. **Outlier probability estimation** for defect detection.
4. **Adaptive thresholding** using a sliding-window statistical test.
5. **Uncertainty-based filtering** for enhanced decision reliability.
6. **Performance evaluation** under fixed/adaptive thresholds and uncertainty gating.

## 📂 Repository Structure

```python
├──baetorch                     # Basic tools
├──uncertainty_ood_v2
├──util
├── dataset/CCD/                # Dataset folder (Replace it with your dataset)
├── result/                     # Detection results and metrics
├── saved_models/                # Trained model checkpoints
├── main.py                      # Main execution script (this code)
└── README.md                    # Documentation
```

## ▶️ Running the Script

```
python main.py
```

The script will:

1. Load and preprocess data.
2. Train two models:
   - Raw CCD image model
   - Wavelet scalogram model
3. Evaluate both under different detection modes.
4. Save results and plots.

✅ The UQAT-BAE method in this project is based on and adapted from [GitHub - bangxiangyong/bae-anomaly-uncertainty: Bayesian autoencoders with uncertainty quantification: Towards trustworthy anomaly detection](https://github.com/bangxiangyong/bae-anomaly-uncertainty)