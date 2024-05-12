import torch
from torch import nn
import numpy as np
from gen_names.config import Params

class Transformer_Encoder(nn.Module):
    """
    Класс трансформера-энкодера, нужен для транспонирования входных данных
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.TransformerEncoder(*args, **kwargs)
    
    def forward(self, src, *args, **kwargs):
        src = src.transpose(0, 1).contiguous()  # Shape = (MaxInLen, BatchSize, EmbSize)
        result = self.net(src, *args, **kwargs)  # Shape = (TargetLen, BatchSize, EmbSize)
        result = result.transpose(0, 1).contiguous()  # Shape = (BatchSize, TargetLen, EmbSize)
        return result

class Transformer_Model(nn.Module):
    def __init__(self, args:Params):
        super().__init__()
        self.embedding_size = args.model.embedding_dim
        self.embeddings = nn.Embedding(args.model.vocab_size, self.embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(args.model.emb_dropout)
        self.backbone = Transformer_Encoder(
            nn.TransformerEncoderLayer(
                d_model = self.embedding_size,
                nhead = args.model.nhead,
                dim_feedforward = args.model.dim_feedforward,
                dropout = args.model.dropout
            ),
            num_layers = args.model.num_layers
        )
        self.out = nn.Linear(self.embedding_size, args.model.vocab_size)
    
    def mask_for_transformer(self, length):
        # Создаём маску единиц
        full_mask = torch.ones(length, length)
        # Создаём диагональную маску с булевыми значениями
        ignore_mask = torch.tril(full_mask) < 1
        # Заполняем False диагональной маски в "маске единиц" значениями -inf
        full_mask.masked_fill_(ignore_mask, float('-inf'))
        # Остальное - нулями
        full_mask.masked_fill_(~ignore_mask, 0)
        return full_mask

    def positional_encoding(self, max_length, embedding_size):
        # Создаём массив, по которому будут генерироваться синусы и косинусы
        time = np.pi * torch.arange(0, max_length).float()
        freq_dividers = torch.arange(1, embedding_size // 2 + 1).float()
        inputs = time[:, None] / freq_dividers[None, :]
        
        # Берём значения синусов и косинусов в качестве ответа
        result = torch.zeros(max_length, embedding_size)
        result[:, 0::2] = torch.sin(inputs)
        result[:, 1::2] = torch.cos(inputs)
        return result
    
    def forward(self, x):
        batch_size, max_in_length = x.shape

        # Создание маски
        seed_padding_mask = x == 0
        dependency_mask = self.mask_for_transformer(max_in_length).to(x.device)
        
        # Эмбеддинг и позиционное кодирование
        seed_embs = self.embeddings(x)  # Shape = (BatchSize, MaxInLen, EmbSize)
        pos_codes = self.positional_encoding(max_in_length, self.embedding_size)
        pos_codes = pos_codes.unsqueeze(0).to(seed_embs.device)
        seed_embs = seed_embs + pos_codes
        seed_embs = self.emb_dropout(seed_embs)

        # Shape =  (BatchSize, TargetLen, EmbSize)
        target_features = self.backbone(
            seed_embs,
            mask=dependency_mask,
            src_key_padding_mask=seed_padding_mask
        )
        logits = self.out(target_features)  # Shape =  (BatchSize, TargetLen, VocabSize)
        return logits