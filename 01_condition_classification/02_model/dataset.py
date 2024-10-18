import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from configures import CFG
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from multiprocessing import Manager
import pandas as pd

class CustomImageDataset(Dataset):
    def __init__(self, data, transforms=None, cache=None):
        self.data = data
        self.transforms = transforms
        # self.num_classes = 4
        self.cache_image = cache
        # self.imgdir = image_dir

    def __len__(self):
        return len(self.data)
        # return len(self.data) * len(self.transforms)

    def __getitem__(self, idx):
        def get_image_cache(idx):
            img_path = self.data.iloc[idx, 1]

            if img_path in self.cache_image:
                return img_path, self.cache_image[img_path]

            else:
                img_array = np.fromfile(img_path, np.uint8)

                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                image = cv2.cvtColor(image, cv2.IMREAD_COLOR)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
                image /= 255.

                if self.transforms is not None:
                    image = self.transforms(image=image)['image']

                self.cache_image[img_path] = torch.as_tensor(image, dtype=torch.float32)
                return img_path, self.cache_image[img_path]

        img_path, image = get_image_cache(idx)
        label = self.data.iloc[idx, 2]
        # label = torch.as_tensor(label, dtype=torch.long)
        # print(type(label))

        return {'image': image, 'label': label, 'path': img_path}


def get_transforms():
    return [
        A.Compose([A.Resize(224, 224), A.HorizontalFlip(p=1), ToTensorV2()]),
        A.Compose([A.Resize(224, 224), A.VerticalFlip(p=1), ToTensorV2()]),
        A.Compose([A.Resize(224, 224), A.HorizontalFlip(p=1), A.VerticalFlip(p=1), ToTensorV2()]),
        A.Compose([A.Resize(224, 224), ToTensorV2()]),
    ]


def transform():
    return A.Compose([
        A.Resize(224,224),
        ToTensorV2()])

def make_loader(data, batch_size, cache, data_type='train'):

    if data_type == 'train':
        transform = A.Compose([
            A.Resize(224, 224),
            ToTensorV2()])
        return DataLoader(
                dataset = CustomImageDataset(data, transforms=transform, cache=cache),
                batch_size = batch_size,
                shuffle = True
            )
    else:
        transform = A.Compose([
        A.Resize(224,224),
        ToTensorV2()])


        return DataLoader(
                dataset = CustomImageDataset(data, transforms=transform, cache=cache),
                batch_size = batch_size,
                shuffle = False
            )



def main():
    manager = Manager()
    img_cache = manager.dict()
    train = pd.read_csv("../../02_classification/train_dataset.csv")
    valid = pd.read_csv("../../02_classification/test_dataset.csv")
    train_loader = make_loader(train, batch_size=CFG['BATCH_SIZE'], cache=img_cache, data_type='train')

    for batch in train_loader:
        images = batch['image']  # 여러 증강된 이미지가 포함됨
        labels = batch['label']
        paths = batch['path']

        print(type(images))


if __name__ == '__main__':
    main()