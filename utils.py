import os
from collections import OrderedDict, namedtuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lr_sheduler import CustomScheduler, PolyScheduler


def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple("GenericDict", dictionary.keys())(**dictionary)


# def save_checkpoint(model, optimizer, scheduler, epoch, outdir, epoch_avg_acc):
#     """Saves checkpoint to disk"""
#     filename = f"model_{epoch:04d}_roc_auc_{epoch_avg_acc:.4f}.pth"
#     directory = outdir
#     filename = os.path.join(directory, filename)
#     weights = model.state_dict()
#     state = OrderedDict(
#         [
#             ("state_dict", weights),
#             ("optimizer", optimizer.state_dict()),
#             ("scheduler", scheduler.state_dict()),
#             ("epoch", epoch),
#         ]
#     )

#     torch.save(state, filename)

def save_checkpoint(model, optimizer, scheduler, epoch, outdir, epoch_avg_acc):
    """Saves checkpoint to disk"""
    filename = f"model_{epoch:04d}_acc_{epoch_avg_acc:.4f}.pth"
    directory = outdir
    filename = os.path.join(directory, filename)
    weights = model.state_dict()
    state = OrderedDict(
        [
            ("state_dict", weights),
            ("optimizer", optimizer.state_dict()),
            ("scheduler", scheduler.state_dict()),
            ("epoch", epoch),
        ]
    )

    torch.save(state, filename)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_optimizer(config, net):
    lr = config.train.learning_rate

    print("Opt: ", config.train.optimizer)

    if config.train.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=config.train.momentum,
            weight_decay=config.train.weight_decay,
        )
    elif config.train.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            net.parameters(), lr=lr, weight_decay=config.train.weight_decay
        )
    elif config.train.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            net.parameters(),
            # [{"params": net.backbone.parameters(), "lr": 0.0001}],
            lr=lr,
            weight_decay=config.train.weight_decay,
        )
    else:
        raise Exception("Unknown type of optimizer: {}".format(config.train.optimizer))
    return optimizer


def get_scheduler(config, optimizer):
    if config.train.lr_schedule.name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.train.n_epoch,
            eta_min=0,
            last_epoch=-1,
        )
    elif config.train.lr_schedule.name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train.lr_schedule.step_size,
            gamma=config.train.lr_schedule.gamma,
        )
    elif config.train.lr_schedule.name == "poly":
        warmup_step = (
            config.dataset.num_of_images
            // config.dataset.batch_size
            * config.train.lr_schedule.warmup_epoch
        )
        total_step = (
            config.dataset.num_of_images
            // config.dataset.batch_size
            * config.train.n_epoch
        )
        scheduler = PolyScheduler(
            optimizer=optimizer,
            base_lr=config.train.learning_rate,
            max_steps=total_step,
            warmup_steps=warmup_step,
            last_epoch=-1,
        )
    elif config.train.lr_schedule.name == "custom":
        scheduler = CustomScheduler(optimizer=optimizer, last_epoch=-1, default_lr=config.train.learning_rate)
    else:
        raise Exception(
            "Unknown type of lr schedule: {}".format(config.train.lr_schedule)
        )
    return scheduler


def get_training_parameters(config, net):
    criterion = torch.nn.CrossEntropyLoss().to("cuda")
    # criterion = torch.nn.CrossEntropyLoss(label_smoothing=config.train.label_smoothing).to("cuda")
    optimizer = get_optimizer(config, net)
    scheduler = get_scheduler(config, optimizer)
    return criterion, optimizer, scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __call__(self):
        return self.val, self.avg


def get_max_bbox(bboxes):
    bbox_sizes = [x[2] * x[3] for x in bboxes]
    max_bbox_index = np.argmax(bbox_sizes)
    return bboxes[max_bbox_index]


class Letterbox(object):
    def __init__(self, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, *kwargs):
        self.new_shape = new_shape
        self.color = color
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup

    def __call__(self, image, *kwargs):
        shape = image.shape  # current shape [width, height]
        if isinstance(self.new_shape, int):
            self.new_shape = (self.new_shape, self.new_shape)

        # Scale ratio (new / old)
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[1] - new_unpad[0], self.new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (self.new_shape[1], self.new_shape[0])
            ratio = self.new_shape[1] / shape[1], self.new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.color)  # add border
        return {"image": image}
