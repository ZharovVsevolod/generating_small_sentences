from typing import Literal
from dataclasses import dataclass

@dataclass
class Model:
    name: str
    vocab_size: int
    embedding_dim: int
    num_layers: int

@dataclass
class MambaModel(Model):
    inner_dim: int
    d_conv: int

@dataclass
class LSTMModel(Model):
    hidden_state: int
    dropout: float

@dataclass
class TransformerModel(Model):
    nhead: int
    dim_feedforward: int
    dropout: float
    emb_dropout: float

@dataclass
class Scheduler:
    name: str

@dataclass
class Scheduler_ReduceOnPlateau(Scheduler):
    patience: int
    factor: float

@dataclass
class Scheduler_OneCycleLR(Scheduler):
    expand_lr: int

@dataclass
class Data:
    data_directory: str
    chunk_size: int

@dataclass
class Training:
    project_name: str
    train_name: str
    seed: int
    epochs: int
    batch: int
    lr: float
    wandb_path: str
    model_path: str
    save_best_of: int
    checkpoint_monitor: str
    early_stopping_patience: int

@dataclass
class Generation:
    mode: Literal["top_k", "top_n"]
    k: int
    n: float
    max_len: int
    max_repeat: int|None
    temperature: float
    alpha: float
    train_reweight: bool
    val_reweight: bool
    beamsize: int

@dataclass
class Params:
    model: Model
    data: Data
    training: Training
    scheduler: Scheduler
    generation: Generation