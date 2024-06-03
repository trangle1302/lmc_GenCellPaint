import cv2
import albumentations as A
import warnings

warnings.filterwarnings("ignore")
import pandas as pd

import torch.nn as nn
from einops import rearrange

from ldm.data import image_processing
from ldm.util import instantiate_from_config

# from ldm.data.serialize import TorchSerializedList
import torch
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split
import numpy as np
import tifffile
from analysis.get_embeddings import rescale_2_98

# from skimage.filters import threshold_local
from PIL import Image

#location_mapping = {"Mitochondria": 0, "Actin": 1, "Tubulin": 2}
#tiff_ch_maping = {"BF": 0, "Nucleus": 1, "Mitochondria": 2, "Actin": 3, "Tubulin": 4}


def load_jump(file_id, chs, rescale=True, sampling=False):
    data_dir = "/scratch/groups/emmalu/JUMP/processed_tiled/JUMP_processed_tiled"
    # JUMP channels  {1: Mito, 2: AGP, 3: NucleoliRNA, 4: ER, 5: Nucleus, 6: BF, 7: BF, 8: BF}
    img_id, tile = file_id.split("_")
    if sampling:
        ch = np.random.choice(chs)
        chs = [ch, ch, ch]
        print(chs)
    imgs = []
    for ch in chs:
        image_path = (
            img_id + f"p01-ch{ch}sk1fk1fl1_{int(tile)}.png"
        )
        imgarray = np.array(Image.open(f"{data_dir}/{image_path}"))
        imgs.append(imgarray)

    full_res_image = np.stack(imgs, axis=2)
    if rescale:
        img_rescale = []
        for ch in range(len(chs)):
            img_ = rescale_2_98(full_res_image[:, :, ch])
            # print(img_.mean(), full_res_image[:,:,ch].mean())
            img_rescale.append(img_)
        img_rescale = np.stack(img_rescale, axis=2)

    return full_res_image


class JUMP:
    def __init__(
        self,
        group="train",
        path_to_metadata=None,
        input_channels=None,
        input_sampling=False,
        output_channels=None,
        size=512,
        scale_factor=1,
        flip_and_rotate=True,
        return_info=False,
    ):
        # Define channels
        self.input_channels = input_channels
        if output_channels is None:
            self.output_channels = input_channels
        else:
            self.output_channels = output_channels
        self.input_sampling = input_sampling
        self.metadata = pd.read_csv(path_to_metadata)
        # print(self.metadata.shape, self.metadata.columns)
        self.metadata = self.metadata[self.metadata.Study=='jump']
        if 'split' not in self.metadata.columns:
            train_data, test_data = train_test_split(
                self.metadata, test_size=0.05, random_state=42
            )
            self.metadata["split"] = [
                "train" if idx in train_data.index else "validation"
                for idx in self.metadata.index
            ]

        self.metadata = self.metadata.sample(frac=1).reset_index(drop=True)
        self.indices = self.metadata[(self.metadata.split == group)].index
        self.image_ids = self.metadata[(self.metadata.split == group)].Id

        self.return_info = return_info

        self.final_size = int(size * scale_factor)
        self.transforms = []
        self.flip_and_rotate = flip_and_rotate
        if self.flip_and_rotate:
            self.transforms.extend(
                [
                    A.RandomRotate90(p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.RandomResizedCrop(
                        height=self.final_size,
                        width=self.final_size,
                        scale=(0.7, 0.95),
                        p=0.5,
                    ),
                ]
            )

        self.transforms.extend(
            [
                A.geometric.resize.Resize(
                    height=self.final_size,
                    width=self.final_size,
                    interpolation=cv2.INTER_LINEAR,
                )
            ]
        )
        self.data_augmentation = A.Compose(self.transforms)
        self.data_augmentation_simple = A.Compose(
            [
                A.RandomRotate90(p=1.0),
                A.HorizontalFlip(p=0.5),
                A.geometric.resize.Resize(
                    height=self.final_size,
                    width=self.final_size,
                    interpolation=cv2.INTER_LINEAR,
                ),
            ]
        )

        print(
            f"Dataset group: {group}, length: {len(self.indices)}, in channels: {self.input_channels},  output channels: {self.output_channels}"
        )

    def __len__(self):
        return len(self.image_ids)
        #return 40 # testing

    def __getitem__(self, i):
        sample = {}
        # get image
        img_index = self.indices[i]
        info = self.metadata.iloc[img_index].to_dict()

        file_id = info["Id"]

        cond = list(map(int, info["organelles"].split(",")))
        cond = torch.tensor(cond[-3:])

        imarray = load_jump(file_id, self.input_channels, rescale=True, sampling=self.input_sampling)
        targetarray = load_jump(file_id, self.output_channels, rescale=True)


        assert image_processing.is_between_0_255(imarray)
        assert image_processing.is_between_0_255(targetarray)

        if (
            info["Study"] != "jump"
        ):  # info['TL'] == 'DIC': #info['Study'] in ['Study_6', 'Study_8', 'Study_9']:
            transformed = self.data_augmentation(image=imarray, mask=targetarray)
        else:
            transformed = self.data_augmentation_simple(image=imarray, mask=targetarray)
        imarray = transformed["image"]
        targetarray = transformed["mask"]
        imarray = (imarray / 255).astype("float32")
        targetarray = (targetarray / 255).astype("float32")
        # print(imarray.dtype, targetarray.dtype)
        imarray = image_processing.convert_to_minus1_1(imarray)
        targetarray = image_processing.convert_to_minus1_1(targetarray)

        #assert imarray.shape == (self.final_size, self.final_size, 3)
        #assert targetarray.shape == (self.final_size, self.final_size, 3)

        
        sample.update(
            {"image": imarray, "ref-image": targetarray, "location_classes": cond}
        )
        if self.return_info:
            sample["info"] = info

        return sample

