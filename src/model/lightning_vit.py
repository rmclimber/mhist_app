from lightning import LightningModule
from transformers import ViTConfig, ViTModel
import torch
import torch.nn as nn

from ..experiment.results_info import *

class LightningViT(LightningModule):
    def __init__(self, 
                 num_classes: int, 
                 pretrained_model_name: str = "google/vit-base-patch16-224",
                 lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.config = ViTConfig.from_pretrained(pretrained_model_name)
        self.vit = ViTModel.from_pretrained(pretrained_model_name, config=self.config)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits

    def step(self, batch, mode: str):
        pixel_values, labels = batch
        logits = self(pixel_values)
        loss = self.loss_fn(logits, labels)
        # Only log if trainer is available (e.g., during actual training)
        try:
            if self.trainer is not None:
                self.log(f"{mode}_loss", loss)
        except RuntimeError:
            # Trainer not attached, skip logging
            pass
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_hat = torch.argmax(logits, dim=1)
        return ResultsInfo(y=y, logits=logits, y_hat=y_hat)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer