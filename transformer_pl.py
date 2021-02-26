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
        # self.norm2 = LayerNorm(d_model)  # skip
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)  # skip
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
        #                            key_padding_mask=memory_key_padding_mask)[0]
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


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


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config, ntokens, d_model=512, nhead=8, num_encoder_layers=0,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", custom_encoder=None, custom_decoder=None):
        super(DecoderOnlyTransformer, self).__init__()
        # model vars
        self.config = config
        self.d_model = d_model
        self.nhead = nhead

        # decoder setup
        decoder_layer = CustomTransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation)
        decoder_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm)

        # embedding setup
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.to_embedding = nn.Embedding(ntokens, d_model)

        # output setup
        self.linear = nn.Linear(d_model, ntokens)

        # training setup
        self.criterion = nn.CrossEntropyLoss()
        self.ntokens = ntokens;

        self._reset_parameters()

    def training_init(self):
        learning_rate = extract_config( self.config, "learning_rate" )

        pass

    def forward(self, tgt, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):

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
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        # return after linear layer
        return self.linear(output)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def training_step(self, batch, batch_idx):
        data, targets = batch
        src_mask = self.generate_square_subsequent_mask(data.size(0))
        output = self.model(data, src_mask)
        loss = self.criterion(output.view(-1, self.ntokens), targets)

        # wandb logging
        wandb.log({
            # "epoch": epoch,
            "batch": batch_idx,
            "batch_loss": loss.item(),
            "ppl": math.exp(loss.item()),
            # "learning_rate": self.scheduler.get_lr()[0],
        })

        return loss

    def validation_step(self, batch):
        data, targets = batch
        src_mask = self.generate_square_subsequent_mask(data.size(0))
        output = self.model(data, src_mask)
        output_flat = output.view(-1, self.ntokens)
        loss = self.criterion(output_flat, targets)

        wandb.log({"val_loss": loss, "val_ppl": math.exp(loss)})
        return loss

    def configure_optimizers(self):
        learning_rate, adam_b1, adam_b2, adam_l2_weightdecay = extract_config(
            self.config, "learning_rate", "adam_b1", "adam_b2", "adam_l2_weightdecay")

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(
            adam_b1, adam_b2), weight_decay=adam_l2_weightdecay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        return [optimizer], [scheduler]
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": self.scheduler
        # }



if __name__ == "__main__":
    # instantiate for testing
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from data_pl import load_data
    from constants import *

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

    model = DecoderOnlyTransformer(config, ntokens)
    trainer = pl.Trainer(gpus=2, accelerator="dp")
    trainer.fit(model, train_loader, val_loader)
