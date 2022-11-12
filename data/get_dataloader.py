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

    print("Preparing train reader...")
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
    print("Done.")

    # print("Preparing valid reader...")
    # hard_val_dataset224 = dataset.ValDataset(val_path=config.dataset.val_path,
    #                                     val_pairs_txt=config.dataset.hard_val_pairs_txt,
    #                                     val_bbox_txt_train=config.dataset.val_bbox_txt_train,
    #                                     val_bbox_txt_val=config.dataset.val_bbox_txt_val,
    #                                     transforms=augmentations.get_val_aug(config))

    # hard_val_dataset232 = dataset.ValDataset(val_path=config.dataset.val_path,
    #                                     val_pairs_txt=config.dataset.hard_val_pairs_txt,
    #                                     val_bbox_txt_train=config.dataset.val_bbox_txt_train,
    #                                     val_bbox_txt_val=config.dataset.val_bbox_txt_val,
    #                                     transforms=augmentations.get_val_aug(config, resize=232))
    # hard_valid_loader224 = torch.utils.data.DataLoader(
    #     hard_val_dataset224,
    #     batch_size=config.dataset.batch_size,
    #     shuffle=False,
    #     num_workers=config.dataset.num_workers,
    #     drop_last=False,
    #     pin_memory=True
    # )
    # hard_valid_loader232 = torch.utils.data.DataLoader(
    #     hard_val_dataset232,
    #     batch_size=config.dataset.batch_size,
    #     shuffle=False,
    #     num_workers=config.dataset.num_workers,
    #     drop_last=False,
    #     pin_memory=True
    # )
    # medium_val_dataset = dataset.ValDataset(val_path=config.dataset.val_path,
    #                                 val_pairs_txt=config.dataset.medium_val_pairs_txt,
    #                                 val_bbox_txt_train=config.dataset.val_bbox_txt_train,
    #                                 val_bbox_txt_val=config.dataset.val_bbox_txt_val,
    #                                 transforms=augmentations.get_val_aug(config))
    # medium_valid_loader = torch.utils.data.DataLoader(
    #     medium_val_dataset,
    #     batch_size=config.dataset.batch_size,
    #     shuffle=False,
    #     num_workers=config.dataset.num_workers,
    #     drop_last=False,
    #     pin_memory=True
    # )

    # print("Done.")
    return train_loader, val_loader
