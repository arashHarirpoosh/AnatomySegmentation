import os
import sys
import torch
import numpy as np
import nibabel as nib
import monai.data as data
from monai.transforms import (
    Compose,
    RepeatChannel,
    SpatialPad,
    SpatialPadd,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    NormalizeIntensity,
    NormalizeIntensityd,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandAffined,
    RandZoomd,
)
from monai.networks.nets import SwinUNETR

spatial_size = (128, 128, 128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_transform(num_samples):
    return Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),

            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),  # ToDo: prob=0.15
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5),
                                approx='sampled', prob=0.2),  # ToDo: prob=0.2
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.5),  # ToDo: prob=0.3
            RandScaleIntensityd(keys=["image"], factors=(0.65, 1.5), prob=0.15),
            RandAffined(keys=["image", "label"],
                        rotate_range=(0.5235987755982988, 0.5235987755982988, 0.5235987755982988),
                        mode=['bilinear', 'nearest'], prob=0.2, device=device),
            RandAffined(keys=["image", "label"], scale_range=(0.3, 0.3, 0.3), mode=['bilinear', 'nearest'], prob=0.2,
                        device=device),
            RandZoomd(keys=["image", "label"], min_zoom=0.5, max_zoom=1, mode=['bilinear', 'nearest'], prob=0.25),

            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandScaleIntensityd(keys=["image"], factors=(0.7, 1.3), prob=0.15),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(2.0, 2.0, 2.0),
                # pixdim=(1.5, 1.5, 1.5),
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(
                spatial_size=spatial_size,
                keys=["image", "label"]
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),

            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
                allow_smaller=True
            ),

            # RandFlipd(
            #    keys=["image", "label"],
            #    spatial_axis=[0],
            #   prob=0.10,
            # ),
            # RandFlipd(
            #    keys=["image", "label"],
            #    spatial_axis=[1],
            #    prob=0.10,
            # ),
            # RandFlipd(
            #    keys=["image", "label"],
            #    spatial_axis=[2],
            #    prob=0.10,
            # ),

            # RandRotate90d(
            #     keys=["image", "label"],
            #     prob=0.15,
            #     max_k=3,
            # ),
            # RandShiftIntensityd(
            #     keys=["image"],
            #     offsets=0.10,
            #     prob=0.50,
            # ),

        ]
    )


def get_train_transform_unet():
    return Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
            #         RandGaussianNoised(keys=["image"], prob=0.50, mean=0.0, std=0.1),

            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 1.5),
                #             pixdim=(3.0, 3.0, 3.0),
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(
                spatial_size=spatial_size,
                keys=["image", "label"]
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=False),

            # RandCropByPosNegLabeld(
            #     keys=["image", "label"],
            #     label_key="label",
            #     spatial_size=(96, 96, 96),
            #     pos=1,
            #     neg=1,
            #     num_samples=num_samples,
            #     image_key="image",
            #     image_threshold=0,
            #     allow_smaller=True
            # ),

            #         RandFlipd(
            #             keys=["image", "label"],
            #             spatial_axis=[0],
            #             prob=0.10,
            #         ),
            #         RandFlipd(
            #             keys=["image", "label"],
            #             spatial_axis=[1],
            #             prob=0.10,
            #         ),
            #         RandFlipd(
            #             keys=["image", "label"],
            #             spatial_axis=[2],
            #             prob=0.10,
            #         ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),

        ]
    )


def get_validation_transform():
    return Compose(
        [
            LoadImaged(keys=["image", "label"], ensure_channel_first=True, image_only=False),
            ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 1.5),
                # pixdim=(3.0, 3.0, 3.0),
                mode=("bilinear", "nearest"),
            ),
            SpatialPadd(
                spatial_size=spatial_size,
                keys=["image", "label"]
            ),
            EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
        ]
    )


def generate_file_path(root_path, label_name):
    all_files = os.listdir(root_path)
    return [{'image': f'{root_path}/{i}/ct.nii.gz',
             'label': f'{root_path}/{i}/segments/{label_name}.nii.gz'
             }
            for i in all_files if os.path.isfile(f'{root_path}/{i}/ct.nii.gz') if
            nib.load(f'{root_path}/{i}/segments/{label_name}.nii.gz').get_fdata(dtype=np.float32).max() > 0]


def filter_collate_fn(data_list, pad_to_shape=spatial_size):
    filtered_data = []
    print(len(data_list))
    for d in data_list:
        print(d)
        if d[0]["label"].max() != 0:
            filtered_data.append(d)

    collated_data = data.pad_list_data_collate(filtered_data, pad_to_shape=pad_to_shape)
    return collated_data
