import os
from torchvision import transforms
from PIL import Image
import random
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {'.png', '.jpg'}  # Use a set for faster lookup

def image_walk(root_dir):
    """
    Collects all images within the root directory and its subdirectories.
    """
    images = []
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                images.append(os.path.join(root, filename))
    return images

class CycleDataset(Dataset):
    """
    Dataset class for loading images from two domains for CycleGAN training or testing.
    """
    def __init__(self, to_train: bool, dataroot: str, scale_size: int, crop_size: int,
                 in_order: bool, **kwargs):
        self.dataroot = dataroot
        mode = 'train' if to_train else 'test'
        self.Xdir = os.path.join(self.dataroot, f"{mode}X")
        self.Ydir = os.path.join(self.dataroot, f"{mode}Y")

        for directory in (self.Xdir, self.Ydir):
            assert os.path.isdir(directory), f"{directory} directory does not exist."

        self.X_images = sorted(image_walk(self.Xdir))
        self.Y_images = sorted(image_walk(self.Ydir))
        self.in_order = in_order

        # Transformations applied to each image
        transformations = [
            transforms.Resize(scale_size, transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3)
        ]
        self.transform = transforms.Compose(transformations)

    def __getitem__(self, index):
        X_img_path = self.X_images[index % len(self.X_images)]
        Y_img_path = self.Y_images[index % len(self.Y_images)] if self.in_order \
                     else random.choice(self.Y_images)

        X_img = Image.open(X_img_path).convert('RGB')
        Y_img = Image.open(Y_img_path).convert('RGB')

        return {
            'X': self.transform(X_img),
            'Y': self.transform(Y_img),
            'X_paths': X_img_path,
            'Y_paths': Y_img_path
        }

    def __len__(self):
        return max(len(self.X_images), len(self.Y_images))

    def both_len(self):
        """
        Returns the number of images in both domains.
        """
        return len(self.X_images), len(self.Y_images)
