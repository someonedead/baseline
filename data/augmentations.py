import albumentations as A
import torch
import torchvision as tv
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode

from utils import Letterbox

# train_transform = A.Compose(
#     [
#         A.SmallestMaxSize(max_size=160),
#         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
#         A.RandomCrop(height=128, width=128),
#         A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
#         A.RandomBrightnessContrast(p=0.5),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ]
# )

# normalize = tv.transforms.Normalize(
#     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
# )

# interpolation = InterpolationMode.BILINEAR

# augs = [
#     tv.transforms.Resize(224),
#     tv.transforms.CenterCrop(224),
#     autoaugment.TrivialAugmentWide(interpolation=interpolation),
#     transforms.PILToTensor(),
#     transforms.ConvertImageDtype(torch.float),
#     normalize,
# ]


def get_train_aug(config):
    padding = config.dataset.padding
    image_size = config.dataset.input_size
    image_size_with_padding = image_size + padding
    if config.dataset.augmentations == "default":
        train_augs = A.Compose(
            [
                A.Resize(width=image_size_with_padding, height=image_size_with_padding),
                A.CenterCrop(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                # A.HueSaturationValue(p=1.0, hue_shift_limit=(-100, 100), sat_shift_limit=(-40, 41), val_shift_limit=(-20, 20)),
                # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    elif config.dataset.augmentations == "letterbox":
        train_augs = A.Compose(
            [
                Letterbox((image_size, image_size)),
                A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                # A.HueSaturationValue(p=1.0, hue_shift_limit=(-100, 100), sat_shift_limit=(-40, 41), val_shift_limit=(-20, 20)),
                # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        # train_augs = tv.transforms.Compose(augs)

        # tv.transforms.Compose([
        #     tv.transforms.RandomResizedCrop(config.dataset.input_size),
        #     tv.transforms.RandomHorizontalFlip(),
        #     tv.transforms.ToTensor(),
        #     normalize
        # ])
    else:
        raise Exception("Unknonw type of augs: {}".format(config.dataset.augmentations))
    return train_augs


def get_val_aug(config, resize=224):
    padding = config.dataset.padding
    image_size = config.dataset.input_size
    image_size_with_padding = image_size + padding
    if config.dataset.augmentations_valid == "default":
        val_augs = A.Compose(
            [
                A.Resize(width=image_size_with_padding, height=image_size_with_padding),
                A.CenterCrop(height=image_size, width=image_size),
                # A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    if config.dataset.augmentations_valid == "letterbox":
        val_augs = A.Compose(
            [
                Letterbox((image_size, image_size)),
                # A.HorizontalFlip(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        # val_augs = transforms.Compose(
        #     [
        #         transforms.Resize(resize, interpolation=interpolation),
        #         transforms.CenterCrop(224),
        #         transforms.PILToTensor(),
        #         transforms.ConvertImageDtype(torch.float),
        #         normalize,
        #     ]
        # )
    else:
        raise Exception("Unknonw type of augs: {}".format(config.dataset.augmentations))
    return val_augs
