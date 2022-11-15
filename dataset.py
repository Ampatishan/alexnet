import os
from torch.utils import data
from torchvision.io import read_image

class ImagenetDataset(data.Dataset):

    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def get_item(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image,label
