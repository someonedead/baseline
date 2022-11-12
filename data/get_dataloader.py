import torch
from . import dataset, augmentations


def get_dataloaders(config):
    """
    Function for creating training and validation dataloaders
    :param config:
    :return:
    """
    print("Preparing train reader...")
    train_dataset = dataset.WhalesDataset(root=config.dataset.root,
                                          annotation_file=config.dataset.train_list,
                                          transforms=augmentations.get_train_aug(config),
                                          is_cropped=config.dataset.cropped)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.dataset.batch_size,
        shuffle=True,
        num_workers=config.dataset.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print("Done.")

    print("Preparing valid reader...")
    if config.train.valmode == 'simple':
        val_dataset = dataset.WhalesDataset(root=config.dataset.root,
                                            annotation_file=config.dataset.train_list,
                                            transforms=augmentations.get_val_aug(config),
                                            is_cropped=config.dataset.cropped)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
            drop_last=True
        )
    elif config.train.valmode == 'pairs':
        val_dataset = dataset.ValDataset(val_path=config.dataset.val_path,
                                         val_pairs=config.dataset.val_pairs,
                                         val_list=config.dataset.val_list,
                                         transforms=augmentations.get_val_aug(config),
                                         is_cropped=config.dataset.cropped)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=True,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
            drop_last=True
        )
    print("Done.")

    return train_loader, val_loader
