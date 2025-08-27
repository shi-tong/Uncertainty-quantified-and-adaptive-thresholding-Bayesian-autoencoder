from torch.utils.data import Dataset, DataLoader

import numpy as np



class SimpleDataset(Dataset):
    """
    支持额外条件变量 c 的简单数据集封装。
    用于 BAE 模型训练，其中输入可以是 x 或 (x, c)，标签 y 可以省略。
    """

    def __init__(self, x, y=None, c=None):
        self.x = x
        self.y = y
        self.c = c

        # 如果没有标签，生成默认索引作为伪标签
        if self.y is None:
            self.y = np.arange(len(self.x))
            self.y_enabled = False
        else:
            self.y_enabled = True

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx].astype(np.float32)
        y = self.y[idx].astype(np.float32)

        if self.c is not None:
            c = self.c[idx].astype(np.float32)
            return {"x": x, "c": c}, y
        else:
            return x, y



def convert_dataloader(x, y=None, c=None, batch_size=100, shuffle=False, drop_last=False):
    return DataLoader(
        SimpleDataset(x, y, c),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
