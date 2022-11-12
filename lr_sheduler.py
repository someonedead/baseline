from torch.optim.lr_scheduler import _LRScheduler


class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, last_epoch=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = 2
        super(PolyScheduler, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch

    def get_warmup_lr(self):
        alpha = float(self.last_epoch) / float(self.warmup_steps)
        return [self.base_lr * alpha for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.warmup_lr_init for _ in self.optimizer.param_groups]
        if self.last_epoch < self.warmup_steps:
            return self.get_warmup_lr()
        else:
            alpha = pow(
                1
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return [self.base_lr * alpha for _ in self.optimizer.param_groups]


class CustomScheduler(_LRScheduler):
    def __init__(
        self, optimizer, epoch2lr=None, default_lr=0.7 * 0.1**5, last_epoch=-1
    ):
        if epoch2lr is None:
            self.epoch2lr = {
                0: 0.00037997181593569487,
                1: 0.00021997181593569487,
                2: 0.00010097181593569487,
                3: 0.00005097181593569487,
                4: 0.00000997181593569487,
                5: 0.00000197181593569487,
                6: 0.00000197181593569487,
                7: 0.00000197181593569487,
                8: 0.00000197181593569487,
                9: 0.00000197181593569487,
                10: 0.00000197181593569487,
            }
        else:
            self.epoch2lr = epoch2lr
        self.default_lr = default_lr
        super(CustomScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        lr = self._compute_lr_from_epoch()

        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return self.base_lrs

    def _compute_lr_from_epoch(self):
        if self.last_epoch in self.epoch2lr:
            return self.epoch2lr[self.last_epoch]
        return self.default_lr
