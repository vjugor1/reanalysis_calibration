from torch import nn
import torch
import pytorch_lightning as pl
from collections import OrderedDict
import torchmetrics

from sklearn import metrics


class WindRNet(nn.Module):
    def __init__(self, args) -> None:
        super(WindRNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=args["in_channels"],
            out_channels=args["out_channels_1"], 
            kernel_size=args["k_size_1"],
            stride=args["stride_1"],
            dilation=args["dilation_1"],
            padding=args["k_size_1"] - 1,
        )
        self.conv2 = nn.Conv2d(  
            in_channels=args["out_channels_1"],
            out_channels=args["out_channels_2"],
            kernel_size=args["k_size_2"],
            stride=args["stride_2"],
            dilation=args["dilation_2"],
            padding=args["k_size_2"] - 1,
        )
        self.maxpool = nn.MaxPool2d(args["maxpool_2"])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(args["fc_size"], 1)
        self.args = args
        self.net = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            self.maxpool,
            self.flatten,
            self.fc,
        ).double()

    def forward(self, X) -> torch.Tensor:
        output = self.net(X)
        return output


class WindRNetPL(pl.LightningModule):
    ## Initialize. Define latent dim, learning rate, and Adam betas
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.net = WindRNet(self.args)

        self.mse = torchmetrics.MeanSquaredError()
        self.r2 = torchmetrics.R2Score()
        self.mape = torchmetrics.MeanAbsolutePercentageError()

        self.loss_f = nn.MSELoss()

    def forward(self, X):
        return self.net(X)

    def loss(
        self, y_hat, y
    ):  
        return self.loss_f(y_hat, y)

    def training_step(self, batch, batch_idx):
        objs, target = batch

        predictions = self(objs)
        loss = self.loss(predictions, target)

        # logging

        self.log("train_loss", loss, prog_bar=True)

        tqdm_dict = {
            "train_loss": loss,
        }
        output = OrderedDict(
            {
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def training_step_end(self, outputs):
        # update and log
        predictions = outputs["preds"]
        target = outputs["target"]
        ms_err = self.mse(predictions, target.view(-1, predictions.shape[-1]))
        map_err = self.mape(predictions, target.view(-1, predictions.shape[-1]))
        r2_sc = self.r2(predictions, target.view(-1, predictions.shape[-1]))

        self.logger.experiment.add_scalars(
            "clf_metrics_train",
            {
                "train_mse": ms_err,
                "train_mape": map_err,
                "train_r2": r2_sc,
            },
            global_step=self.global_step,
        )

    def validation_step(self, batch, batch_idx):
        objs, target = batch

        predictions = self(objs)
        loss = self.loss(predictions, target)

        self.log("val_loss", loss, prog_bar=True)
        tqdm_dict = {"val_loss": loss}
        output = OrderedDict(
            {
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def validation_step_end(self, outputs):
        # update and log
        predictions = outputs["preds"]
        target = outputs["target"]
        ms_err = self.mse(predictions, target.view(-1, predictions.shape[-1]))
        map_err = self.mape(predictions, target.view(-1, predictions.shape[-1]))
        r2_sc = self.r2(predictions, target.view(-1, predictions.shape[-1]))

        self.logger.experiment.add_scalars(
            "clf_metrics_val",
            {
                "val_mse": ms_err,
                "val_mape": map_err,
                "val_r2": r2_sc,
            },
            global_step=self.global_step,
        )
        self.log("val_r2", r2_sc)

    def test_step(self, batch, batch_idx):
        objs, target = batch

        predictions = self(objs)
        loss = self.loss(predictions, target)

        self.log("test_loss", loss, prog_bar=True)
        tqdm_dict = {"test_loss": loss}
        output = OrderedDict(
            {
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
                "preds": predictions,
                "target": target,
            }
        )
        return output

    def test_step_end(self, outputs):
        # update and log
        predictions = outputs["preds"]
        target = outputs["target"]
        ms_err = self.mse(predictions, target.view(-1, predictions.shape[-1]))
        map_err = self.mape(predictions, target.view(-1, predictions.shape[-1]))
        r2_sc = self.r2(predictions, target.view(-1, predictions.shape[-1]))

        self.logger.experiment.add_scalars(
            "clf_metrics_test",
            {
                "test_mse": ms_err,
                "test_mape": map_err,
                "test_r2": r2_sc,
            },
            global_step=self.global_step,
        )

    def configure_optimizers(self):
        lr = self.args["lr"]
        b1 = self.args["b1"]
        b2 = self.args["b2"]

        opt = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(b1, b2))
        return [opt], []
