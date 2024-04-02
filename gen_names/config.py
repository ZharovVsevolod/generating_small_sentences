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
class Params:
    model: Model
    data: Data
    training: Training
    scheduler: Scheduler