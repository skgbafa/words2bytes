# Decoder only transformer implementation
import math
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import TransformerDecoder, LayerNorm
from typing import Optional, Any

import pytorch_lightning as pl

from utils import extract_config
from constants import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Decoder Only implmentation without memory for encoder
# Adapted from pytorch implmentation @ https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(CustomTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor= None, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))

# decoder only implmentation
# pytorch implmentation for torch ligthning
# class Transformer(pl.LightningModule):

# global counter
training_tokens_processed = 0


class DecoderOnlyTransformer(pl.LightningModule):
    training_tokens_processed = 0

    def __init__(self, config, ntokens, trainer=None, activation="relu"):
        super(DecoderOnlyTransformer, self).__init__()
        
        # model vars
        self.extract_config(config)
        self.ntokens = ntokens
        self.trainer = trainer

        # decoder setup
        decoder_layer = CustomTransformerDecoderLayer(
            self.d_model, self.n_heads, self.ff_dimension, self.dropout, activation)
        decoder_norm = LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, self.n_decoder_layers, decoder_norm)

        # embedding setup
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        self.to_embedding = nn.Embedding(ntokens, self.d_model)

        # output setup
        self.linear = nn.Linear(self.d_model, ntokens)

        # training setup
        self.criterion = nn.CrossEntropyLoss()
        
        self._reset_parameters()

    def extract_config(self, config):
        embedding_dimension, n_attention_heads, n_decoder_layers, ff_dimension, dropout, learning_rate, adam_b1, adam_b2, adam_l2_weightdecay, gamma, enable_lr_scheduler, T_max = extract_config(
            config, "embedding_dimension", "n_attention_heads", "n_decoder_layers", "ff_dimension", "dropout", "learning_rate", "adam_b1", "adam_b2", "adam_l2_weightdecay", "gamma", "enable_lr_scheduler", "T_max")

        self.d_model = embedding_dimension
        self.n_heads = n_attention_heads
        self.n_decoder_layers = n_decoder_layers
        self.ff_dimension = ff_dimension
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.adam_b1 = adam_b1
        self.adam_b2 = adam_b2
        self.gamma = gamma
        self.adam_l2_weightdecay = adam_l2_weightdecay
        self.enable_lr_scheduler = enable_lr_scheduler
        self.T_max = T_max

    def forward(self, tgt, tgt_mask=None, tgt_key_padding_mask=None):

        # convert input/targets to embeddings
        tgt = self.to_embedding(tgt) * math.sqrt(self.d_model)

        # add positional encodings
        tgt = self.pos_encoder(tgt)

        # pytorch checks
        # https://pytorch.org/docs/master/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward
        if tgt.size(2) != self.d_model:
            raise RuntimeError(
                "the feature number of tgt must be equal to d_model")

        # decoder pass
        output = self.decoder(tgt, memory=None, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask)
        # return after linear layer
        return self.linear(output)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=self.device))
                == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        DecoderOnlyTransformer.training_tokens_processed = 0
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def training_step(self, batch, batch_idx):
        data, targets = batch
        src_mask = self.generate_square_subsequent_mask(data.size(0))
        output = self(data, src_mask)
        output_flat = output.view(-1, self.ntokens)
        loss = self.criterion(output_flat, targets)
        self.updateTokenCount(torch.numel(data))

        self.log('training_tokens_processed', DecoderOnlyTransformer.training_tokens_processed)
        self.log('batch_loss', loss.item(), on_step=True, on_epoch=False)
        self.log('avg_loss', loss.item(), on_step=False, on_epoch=True)
        self.log('batch_ppl', math.exp(loss.item()), on_step=True, on_epoch=False)
        self.log('avg_ppl', math.exp(loss.item()), on_step=False, on_epoch=True)
        # self.log('learning_rate', self.scheduler.get_last_lr()[0], on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        src_mask = self.generate_square_subsequent_mask(data.size(0))
        output = self(data, src_mask)
        output_flat = output.view(-1, self.ntokens)
        loss = self.criterion(output_flat, targets)

        self.log('val_avg_loss', loss.item(), on_step=False, on_epoch=True)
        self.log('val_avg_ppl', math.exp(loss.item()), on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        data, targets = batch
        src_mask = self.generate_square_subsequent_mask(data.size(0))
        output = self(data, src_mask)
        output_flat = output.view(-1, self.ntokens)
        loss = self.criterion(output_flat, targets)

        self.log('test_avg_loss', loss.item(), on_step=False, on_epoch=True)
        self.log('test_avg_ppl', math.exp(loss.item()), on_step=False, on_epoch=True)

        return loss

    def updateTokenCount(self, count):
        DecoderOnlyTransformer.training_tokens_processed += count

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(
            self.adam_b1, self.adam_b2), weight_decay=self.adam_l2_weightdecay)

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, 5000, gamma=self.gamma),
            'name': 'lr_scheduler',
            'interval': 'step',
            'frequency': 1,
        }

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        },

def logTensor(tensor, note: None):
    try:
        print(note, tensor.shape, tensor.get_device())
    except:
        print(note, tensor.shape)


if __name__ == "__main__":
    # instantiate for testing
    from data import load_data

    config = {
        "embedding_dimension": 200,
        "ff_dimension": 200,
        "n_attention_heads": 2,
        "n_encoder_layers": 0,
        "n_decoder_layers": 2,
        "dataset": Dataset.PennTreebank.name,
        "segmentation": Segmentation.Word.name,
        "max_seq_len": 35,
        "batch_size": 20,
        "eval_batch_size": 10,
        "dropout": 0.2,
        "n_epochs": 3,
        "learning_rate": 0.0001,
        "adam_b1": 0.9,
        "adam_b2": 0.999,
        "adam_l2_weightdecay": 0.01,
        "loss_criterion": "CrossEntropyLoss"
    }

    train_loader, val_loader, test_loader, vocab = load_data(config)
    ntokens = len(vocab.stoi)

    trainer = pl.Trainer(gpus=4, accelerator="ddp")
    model = DecoderOnlyTransformer(config, ntokens, trainer)
    trainer.fit(model, train_loader, val_loader)
