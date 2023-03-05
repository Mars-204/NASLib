## Generating Imagenet 16 -120 C
import cv2

import os
import os.path

import numpy as np
import torch
import torchvision.transforms as transforms

from naslib.utils.DownsampledImageNet import ImageNet16
from imagenet_c import corruption_tuple


class ImagenetC(torch.utils.data.Dataset):
    def __init__(self, dataset,corruption,sev):
        self.dataset = dataset
        self.corruption = corruption
        self.sev = sev

    def __getitem__(self, i):
        x, y = self.dataset[i]
        x = corrupt(x,self.corruption,self.sev)
        return x, y

    def __len__(self):
        return len(self.dataset)

def corrupt(image,corruption,sev):
    image = corruption(image,sev)
    return image

def imagenet_c(dataset):
    corruption = corruption_tuple
    dir = '/work/ws-tmp/g059997-NASLIB/g059997-naslib-1675210204/g059997-naslib-1667607005/NASLib_mod/naslib/data/imagenet_c'
    for c in corruption:
        print(c)
        for s in range(1,6):
            aug_data = '_{}_{}_'.format(c,s)
            dataset = ImagenetC(dataset=dataset,corruption=c,sev=s)
            new_path = os.path.join(dir,aug_data)
            cv2.imwrite(new_path,dataset)
    
    return dataset


def main():
    from naslib.utils.DownsampledImageNet import ImageNet16
    config = config.evaluation
    data = '/work/ws-tmp/g059997-NASLIB/g059997-naslib-1675210204/g059997-naslib-1667607005/NASLib_mod/naslib/data/ImageNet16-120'
    dataset = 'ImageNet_16_120'
    IMAGENET16_MEAN = [x / 255 for x in [122.68, 116.66, 104.01]]
    IMAGENET16_STD = [x / 255 for x in [63.22, 61.26, 65.09]]
    valid_transform = valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET16_MEAN, IMAGENET16_STD),
        ]
    )
    data_folder = f"{data}/{dataset}"
    # train_data = ImageNet16(
    #     root=data_folder,
    #     train=True,
    #     transform=train_transform,
    #     use_num_of_class_only=120,
    # )
    test_data = ImageNet16(
        root=data_folder,
        train=False,
        transform=valid_transform,
        use_num_of_class_only=120,
    ) 
    test_data = imagenet_c(test_data)
