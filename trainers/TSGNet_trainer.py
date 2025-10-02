import lightning.pytorch as pl
import torch
import torch.nn as nn
from losses import *
from metrics import calculate_psnr, calculate_ssim
from models.TSGNet import Model

class TSGNetTrainer(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=0.0, max_epochs=100):
        super(TSGNetTrainer, self).__init__()
        self.save_hyperparameters()
        self.model = Model()
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):

        img0, img1, img2, embt = batch
        imgt_pred, loss_rec, loss_geo, loss_dis = self.model(img0, img2, embt, img1)
        loss = loss_rec + loss_geo + loss_dis

        psnr = calculate_psnr(imgt_pred, img1)
        ssim = calculate_ssim(imgt_pred, img1)

        self.log("Loss/train", loss, sync_dist=True)
        self.log("Recon_Loss/train", loss_rec, sync_dist=True)
        self.log("Geo_Loss/train", loss_geo, sync_dist=True)
        self.log("PSNR/train", psnr, sync_dist=True)
        self.log("SSIM/train", ssim, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):

        img0, img1, img2, embt = batch
        imgt_pred, loss_rec, loss_geo, loss_dis = self.model(img0, img2, embt, img1)
        loss = loss_rec + loss_geo + loss_dis

        psnr = calculate_psnr(imgt_pred, img1)
        ssim = calculate_ssim(imgt_pred, img1)

        self.log("Loss/test", loss, sync_dist=True)
        self.log("Recon_Loss/test", loss_rec, sync_dist=True)
        self.log("Geo_Loss/test", loss_geo, sync_dist=True)
        self.log("PSNR/test", psnr, sync_dist=True)
        self.log("SSIM/test", ssim, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.max_epochs)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]