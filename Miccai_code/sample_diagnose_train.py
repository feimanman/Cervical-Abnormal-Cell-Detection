# @Author: yican, yelanlan
# @Date: 2020-06-16 20:36:19
# @Last Modified by:   yican.yc
# @Last Modified time: 2020-06-16 20:36:19
# Standard libraries
import os
import gc
from time import time
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Third party libraries
import torch
from dataset1 import generate_transforms, generate_dataloaders
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import KFold

# User defined libraries
from fenleimodels import se_resnext50_32x4d
from utils_sample_diagnose import init_hparams, init_logger, seed_reproducer, load_data
from loss_function import CrossEntropyLossOneHot
from lrs_scheduler import WarmRestart, warm_restart


class CoolSystem(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # 让每次模型初始化一致, 不让只要中间有再次初始化的情况, 结果立马跑偏
        seed_reproducer(self.hparams.seed)

        self.model= se_resnext50_32x4d()
        self.criterion = CrossEntropyLossOneHot()
        self.logger_kun = init_logger("kun_in", hparams.log_dir)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.scheduler = WarmRestart(self.optimizer, T_max=10, T_mult=1, eta_min=1e-5)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time = batch

        scores = self(images)
        loss = self.criterion(scores, labels)
        # self.logger_kun.info(f"loss : {loss.item()}")
        # ! can only return scalar tensor in training_step
        # must return key -> loss
        # optional return key -> progress_bar optional (MUST ALL BE TENSORS)
        # optional return key -> log optional (MUST ALL BE TENSORS)
        data_load_time = torch.sum(data_load_time)

        return {
            "loss": loss,
            "data_load_time": data_load_time,
            "batch_run_time": torch.Tensor([time() - step_start_time + data_load_time]).to(data_load_time.device),
        }

    def training_epoch_end(self, outputs):
        # outputs is the return of training_step
        train_loss_mean = torch.stack([output["loss"] for output in outputs]).mean()
        self.data_load_times = torch.stack([output["data_load_time"] for output in outputs]).sum()
        self.batch_run_times = torch.stack([output["batch_run_time"] for output in outputs]).sum()

        self.current_epoch += 1
        if self.current_epoch < (self.trainer.max_epochs - 4):
            self.scheduler = warm_restart(self.scheduler, T_mult=2)

        return {"train_loss": train_loss_mean}

    def validation_step(self, batch, batch_idx):
        step_start_time = time()
        images, labels, data_load_time = batch
        data_load_time = torch.sum(data_load_time)
        scores = self(images)
        loss = self.criterion(scores, labels)

        # must return key -> val_loss
        return {
            "val_loss": loss,
            "scores": scores,
            "labels": labels,
            "data_load_time": data_load_time,
            "batch_run_time": torch.Tensor([time() - step_start_time + data_load_time]).to(data_load_time.device),
        }

    def validation_epoch_end(self, outputs):
        # compute loss
        val_loss_mean = torch.stack([output["val_loss"] for output in outputs]).mean()
        self.data_load_times = torch.stack([output["data_load_time"] for output in outputs]).sum()
        self.batch_run_times = torch.stack([output["batch_run_time"] for output in outputs]).sum()

        # compute roc_auc
        scores_all = torch.cat([output["scores"] for output in outputs]).cpu()
        labels_all = torch.round(torch.cat([output["labels"] for output in outputs]).cpu())

        val_roc_auc = roc_auc_score(labels_all, scores_all)
        print(val_roc_auc)
        scores_all = torch.softmax(scores_all,dim=1)
        val_roc_auc = roc_auc_score(labels_all, scores_all)
        print(val_roc_auc)
        # fpr = dict()
        # tpr = dict()
        roc_auc = dict()
        for i in range(labels_all.size()[1]):
            # fpr[i], tpr[i], _ = roc_curve(labels_all[:, i], scores_all[:, i])
            # roc_auc[i] = auc(fpr[i], tpr[i])
            roc_auc[i] = roc_auc_score(labels_all[:, i], scores_all[:, i])
        print(roc_auc)
        # terminal logs
        self.logger_kun.info(
            f"{self.hparams.fold_i}-{self.current_epoch} | "
            f"lr : {self.scheduler.get_lr()[0]:.6f} | "
            f"val_loss : {val_loss_mean:.4f} | "
            f"val_roc_auc : {val_roc_auc:.4f} | "
            f"data_load_times : {self.data_load_times:.2f} | "
            f"batch_run_times : {self.batch_run_times:.2f}"
        )
        # f"data_load_times : {self.data_load_times:.2f} | "
        # f"batch_run_times : {self.batch_run_times:.2f}"
        # must return key -> val_loss
        return {"val_loss": val_loss_mean, "val_roc_auc": val_roc_auc}

