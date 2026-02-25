import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import gzip
import numpy as np

LABELS_CHAR_MAP = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, # digits 0-9
                   65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, # uppercase A-Z
                   97, 98, 100, 101, 102, 103, 104, 110, 113, 114, 116] # lowercase a, b, d, e, f, g, h, n, q, r, t

class EMNISTDataset(Dataset):
    def __init__(self, images_path, labels_path, transforms=None):
        self.images = get_images(images_path) # emnist-balanced-[train/test]-images-idx3-ubyte.gz
        self.labels = get_labels(labels_path) # emnist-balanced-[train/test]-labels-idx1-ubyte.gz
        self.classes = [chr(l) for l in LABELS_CHAR_MAP]
        self.transforms = transforms
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = self.transforms(image)
        target = self.labels[idx]
        return image, target

def get_images(path):
    with gzip.open(path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        images = [F.to_pil_image(image.T) for image in images]
        return images


def get_labels(path):
    with gzip.open(path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return torch.tensor(labels, dtype=torch.int64)
