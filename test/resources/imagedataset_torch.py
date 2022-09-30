import os

from torch.utils.data import Dataset

from torchvision.io import read_image
from torchvision.transforms import Compose, Resize


def resource(filename):
    current_dir = os.path.dirname(__file__)
    resource_dir = os.path.abspath(current_dir)
    return os.path.join(resource_dir, filename)


class TrainImageDataset(Dataset):
    def __init__(self):
        self.images = [
            resource('flowers17/train/0/image_0006.jpg'),
            resource('flowers17/train/1/image_0141.jpg'),
            resource('flowers17/train/2/image_0260.jpg'),
            resource('flowers17/train/3/image_0500.jpg'),
        ]
        self.labels = [0, 1, 2, 3]
        self.transform = Compose([Resize((224, 224))])

    def __len__(self): return 4

    def __getitem__(self, item):
        image = read_image(self.images[item])
        label = self.labels[item]
        return self.transform(image).float(), label


class TestImageDataset(Dataset):
    def __init__(self):
        self.images = [
            resource('flowers17/test/0/image_0006.jpg'),
            resource('flowers17/test/1/image_0141.jpg'),
            resource('flowers17/test/2/image_0260.jpg'),
            resource('flowers17/test/3/image_0500.jpg'),
        ]
        self.labels = [0, 1, 2, 3]
        self.transform = Compose([Resize((224, 224))])

    def __len__(self): return 4

    def __getitem__(self, item):
        image = read_image(self.images[item])
        label = self.labels[item]
        return self.transform(image).float(), label
