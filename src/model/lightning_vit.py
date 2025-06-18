from lightning import LightningModule
from transformers import ViTConfig, ViTModel
import torch
import torch.nn as nn

class LightningViT(LightningModule):
    def __init__(self, num_classes, pretrained_model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.num_classes = num_classes
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
        self.log(f"{mode}_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer