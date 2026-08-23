"""
PyTorch Dataset for the labeled ECG image classification task
(normal vs abnormal), built from labels.csv produced by
prepare_image_labels.py.
"""
from PIL import Image
from torch.utils.data import Dataset


class ECGImageDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        df: pandas DataFrame with columns ['image_path', 'label']
            (a train/val/test split slice of labels.csv)
        transform: torchvision transform pipeline
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        return img, label
