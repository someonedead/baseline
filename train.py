import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

from utils import AverageMeter


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          config, epoch) -> None:
    """
    Model training function for one epoch
    :param model: model architecture
    :param train_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param optimizer: selected optimizer for updating weights
    :param config: train process configuration
    :param epoch (int): epoch number
    :return: None
    """
    model.train()
    use_fp16 = config.train.fp16
    if use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    loss_stat = AverageMeter('Loss')
    acc_stat = AverageMeter('Acc.')

    train_iter = tqdm(train_loader, desc=f'Train epoch {epoch}/{config.train.n_epoch-1}', dynamic_ncols=True)

    for step, (x, y) in enumerate(train_iter):
        # out = model(x.cuda().to(memory_format=torch.contiguous_format))
        if use_fp16:
            with torch.cuda.amp.autocast():
                out = model(x.cuda().to(memory_format=torch.contiguous_format), y.cuda())
                # out = model(x.cuda().to(memory_format=torch.contiguous_format))
                loss = criterion(out, y.cuda())
            num_of_samples = x.shape[0]
            loss_stat.update(loss.detach().cpu().item(), num_of_samples)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            # out = model(x.cuda().to(memory_format=torch.contiguous_format))
            out = model(x.cuda().to(memory_format=torch.contiguous_format), y.cuda())
            loss = criterion(out, y.cuda())
            num_of_samples = x.shape[0]

            loss_stat.update(loss.detach().cpu().item(), num_of_samples)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scores = torch.softmax(out, dim=1).detach().cpu().numpy()
        predict = np.argmax(scores, axis=1)
        gt = y.detach().cpu().numpy()

        acc = np.mean(gt == predict)
        acc_stat.update(acc, num_of_samples)

        if step % config.train.freq_vis == 0 and not step == 0:
            acc_val, acc_avg = acc_stat()
            loss_val, loss_avg = loss_stat()
            lr = get_lr(optimizer=optimizer)
            print('Epoch: {}; step: {}; loss: {:.4f}; acc: {:.2f}; lr: {}'.format(epoch, step, loss_avg, acc_avg, lr))

    acc_val, acc_avg = acc_stat()
    loss_val, loss_avg = loss_stat()
    print('Train process of epoch: {} is done; \n loss: {:.4f}; acc: {:.2f}'.format(epoch, loss_avg, acc_avg))


def calc_roc_auc(val_loader, model):
    val_iter = tqdm(val_loader, desc='Val', dynamic_ncols=True, position=2)

    for step, (img1, img2, score) in enumerate(val_iter):
        embeddings1 = model(img1.cuda().to(
            memory_format=torch.contiguous_format)
        ).cpu().numpy()
        embeddings2 = model(img2.cuda().to(
            memory_format=torch.contiguous_format)
        ).cpu().numpy()
        score = score.cpu().numpy()

        if step == 0:
            total_embeddings1 = embeddings1
            total_embeddings2 = embeddings2
            total_scores = score
        else:
            total_embeddings1 = np.vstack((total_embeddings1, embeddings1))
            total_embeddings2 = np.vstack((total_embeddings2, embeddings2))
            total_scores = np.hstack((total_scores, score))

    # total_embeddings1 = normalize(total_embeddings1)
    # total_embeddings2 = normalize(total_embeddings2)
    dists = np.linalg.norm(total_embeddings1 - total_embeddings2, axis=1)
    norm_dists = (2 - dists) / 2
    roc_auc = roc_auc_score(total_scores, norm_dists)
    return roc_auc


def validation_new(model: torch.nn.Module,
                   medium_val_loader: torch.utils.data.DataLoader,
                   hard_val_loader224: torch.utils.data.DataLoader,
                   hard_val_loader232: torch.utils.data.DataLoader,
                   criterion: torch.nn.Module,
                   epoch):
    """
    Model validation function for one epoch
    :param model: model architecture
    :param val_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param epoch (int): epoch number
    :return: float: avg acc
     """

    with torch.no_grad():
        model.eval()
        # medium_roc_auc = calc_roc_auc(medium_val_loader, model)
        medium_roc_auc = 0

        # print(f'ROC_AUC on MEDIUM of epoch {epoch}: {medium_roc_auc:.4f}')
        hard_roc_auc224 = calc_roc_auc(hard_val_loader224, model)
        print(f'ROC_AUC on HARD of epoch {epoch}: {hard_roc_auc224:.4f}')
        hard_roc_auc232 = 0
        # print(f'ROC_AUC on HARD 232 of epoch {epoch}: {hard_roc_auc232:.4f}')
        return medium_roc_auc, hard_roc_auc224, hard_roc_auc232


def validation(model: torch.nn.Module,
               val_loader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               epoch) -> None:
    """
    Model validation function for one epoch
    :param model: model architecture
    :param val_loader: dataloader for batch generation
    :param criterion: selected criterion for calculating the loss function
    :param epoch (int): epoch number
    :return: float: avg acc
     """
    loss_stat = AverageMeter('Loss')
    acc_stat = AverageMeter('Acc.')

    with torch.no_grad():
        model.eval()
        val_iter = tqdm(val_loader, desc='Val', dynamic_ncols=True, position=2)

        for step, (x, y) in enumerate(val_iter):
            # out = model(x.cuda().to(memory_format=torch.contiguous_format))
            out = model(x.cuda().to(memory_format=torch.contiguous_format), y.cuda())
            loss = criterion(out, y.cuda())
            num_of_samples = x.shape[0]

            loss_stat.update(loss.detach().cpu().item(), num_of_samples)

            scores = torch.softmax(out, dim=1).detach().cpu().numpy()
            predict = np.argmax(scores, axis=1)
            gt = y.detach().cpu().numpy()

            acc = np.mean(gt == predict)
            acc_stat.update(acc, num_of_samples)

        acc_val, acc_avg = acc_stat()
        loss_val, loss_avg = loss_stat()
        print('Validation of epoch: {} is done; \n loss: {:.4f}; acc: {:.2f}'.format(epoch, loss_avg, acc_avg))
        return acc_avg
