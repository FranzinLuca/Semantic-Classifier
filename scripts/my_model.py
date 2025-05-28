import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from utils import *


class claimClass(pl.LightningModule):
    def __init__(self, vocab_size_s, vocab_size_c,embedding_dim=100, mode='mean', le = 1e-3, batch_size=128, num_classes=3, dropout=0.4):
        super().__init__()
        self.save_hyperparameters()
        self.embeddingS = nn.Embedding(
            num_embeddings=self.hparams.vocab_size_s,
            embedding_dim=self.hparams.embedding_dim
        )
        
        self.embeddingC = nn.Embedding(
            num_embeddings=self.hparams.vocab_size_c,
            embedding_dim=self.hparams.embedding_dim
        )
        
        self.sequential = nn.Sequential(
            nn.Linear(self.hparams.embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(64, self.hparams.num_classes),
        )
        
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )
        self.precision = torchmetrics.Precision(
            task="multiclass", num_classes=self.hparams.num_classes, average="macro"
        )
        self.recall = torchmetrics.Recall(
            task="multiclass", num_classes=self.hparams.num_classes, average="macro"
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, statements_tensor, claim_tensor, offset_tensor):
        
        embeddedS = self.embeddingS(statements_tensor)
        embeddedC = self.embeddingC(claim_tensor)
        embeddedUnited = torch.cat((embeddedS,embeddedC),dim=1)
        vector_concat = torch.nn.functional.embedding_bag(
            input=torch.arange(embeddedUnited.size(0), device=embeddedUnited.device),
            weight=embeddedUnited,
            offsets=offset_tensor,
            mode=self.hparams.mode
        )
        output = self.sequential(vector_concat)
        return output
    
    def _common_step(self, batch):
        statements_tensor, claim_tensor, offset_tensor, target = batch
        predictions = self(statements_tensor, claim_tensor, offset_tensor)  
        loss = self.criterion(predictions, target)
        accuracy = self.accuracy(predictions, target)
        precision = self.precision(predictions, target)
        recall = self.recall(predictions, target)
        f1_score = self.f1_score(predictions, target)
        return loss, accuracy, precision, recall, f1_score
    
    
    def training_step(self, batch, batch_idx):
        loss, accuracy, precision, recall, f1_score = self._common_step(batch)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": accuracy,
                "train_precision": precision,
                "train_recall": recall,
                "train_f1_score": f1_score,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size
        )
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy, precision, recall, f1_score = self._common_step(batch)
        self.log_dict(
            {
                "validation_loss": loss,
                "validation_accuracy": accuracy,
                "validation_precision": precision,
                "validation_recall": recall,
                "validation_f1_score": f1_score,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size
        )
        return {"loss": loss}
    
    def test_step(self, batch, batch_idx):
        loss, accuracy, precision, recall, f1_score = self._common_step(batch)
        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1_score,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.hparams.batch_size
        )
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.le, weight_decay=1e-4)