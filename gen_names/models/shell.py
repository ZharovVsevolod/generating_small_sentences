import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from typing import Literal
from torchmetrics.text import BLEUScore
from gen_names.config import Params
from gen_names.data import CharTokenizer

from gen_names.models.true_model import Mamba as MB_true
from gen_names.models.true_model import ModelArgs
from gen_names.models.mamba import Mamba
from gen_names.models.lstm import LSTM

from gen_names.generators import BeamGenerator

def conversion_args_to_true_mamba(args: Params):
    true_args = ModelArgs(
        vocab_size = args.model.vocab_size,
        d_model = args.model.embedding_dim,
        n_layer = args.model.num_layers,
        expand = int(args.model.inner_dim / args.model.embedding_dim),
        d_conv = args.model.d_conv
    )
    return true_args

class Model_Lightning_Shell(L.LightningModule):
    def __init__(
            self,
            args: Params,
            tokenizer: CharTokenizer | None = None,
            pad_value: int = 0
        ) -> None:
        super().__init__()

        # Match model that we need
        match args.model.name:
            case "mamba":
                self.inner_model = Mamba(args)
            case "true_mamba":
                args_true = conversion_args_to_true_mamba(args)
                self.inner_model = MB_true(args_true)
            case "lstm":
                self.inner_model = LSTM(args)
            case "transformer":
                print("Support of Transformer model will be added in the future")
                assert True, "Transformer is not supported now"

        self.metric = BLEUScore()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = CharTokenizer()

        self.lr = args.training.lr
        self.pad_value = pad_value

        #-----
        self.args = args

        self.save_hyperparameters()
    
    def forward(self, x) -> torch.Any:
        return self.inner_model(x)
    
    def loss(self, y, y_hat):
        y_flat = y.view(-1, y.shape[-1])  # BatchSize*TargetLen x VocabSize
        y_hat_flat = y_hat.view(-1)  # BatchSize*TargetLen
        actual_loss = F.cross_entropy(y_flat, y_hat_flat, ignore_index=self.pad_value)
        return actual_loss
    
    def compute_bleu_score(self, y, y_hat):
        model_answer = torch.argmax(y, dim=-1).cpu().numpy()
        model_answer = [self.tokenizer.decode(name_idx) for name_idx in model_answer]

        names_target = [self.tokenizer.decode(name_idx.cpu().numpy()) for name_idx in y_hat]

        bleu_score = self.metric(model_answer, names_target)
        return bleu_score
    
    def reweight(self, original):
        alpha = self.args.generation.alpha
        temperature = self.args.generation.temperature

        # Если есть параметр альфа, его применяем по формуле
        if alpha != 0:
            original = torch.mul(original, (1 - alpha)) + alpha / len(original)
        # Делим логарифм весов на температуру для усреднения весов, сила которого зависит от температуры
        distribution = torch.log(original) / temperature

        return distribution

    def lr_scheduler(self, optimizer):
        if self.args.scheduler.name == "ReduceOnPlateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                patience = self.args.scheduler.patience, 
                factor = self.args.scheduler.factor
            )
            scheduler_out = {"scheduler": sched, "monitor": "val_loss"}
        
        if self.args.scheduler.name == "OneCycleLR":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr = self.lr * self.args.scheduler.expand_lr, 
                total_steps = self.args.training.epochs
            )
            scheduler_out = {"scheduler": sched}
        
        return scheduler_out
    
    def training_step(self, batch) -> STEP_OUTPUT:
        x, y_hat = batch

        y = self(x)

        #--------------------------------------------------
        # Reweight section
        if self.args.generation.train_reweight:
            y = self.reweight(y)
        #--------------------------------------------------

        answer_loss = self.loss(y, y_hat)
        bleu_score = self.compute_bleu_score(y, y_hat)

        self.log("train_loss", answer_loss)
        self.log("train_bleu", bleu_score)
    
    def validation_step(self, batch) -> STEP_OUTPUT:
        x, y_hat = batch

        y = self(x)

        #--------------------------------------------------
        # Reweight section
        if self.args.generation.val_reweight:
            y = self.reweight(y)
        #--------------------------------------------------

        answer_loss = self.loss(y, y_hat)
        bleu_score = self.compute_bleu_score(y, y_hat)

        self.log("val_loss", answer_loss)
        self.log("val_bleu", bleu_score)
    
    def test_step(self, batch) -> STEP_OUTPUT:
        x, y_hat = batch

        y = self(x)

        #--------------------------------------------------
        # Reweight section
        if self.args.generation.val_reweight:
            y = self.reweight(y)
        #--------------------------------------------------

        answer_loss = self.loss(y, y_hat)
        bleu_score = self.compute_bleu_score(y, y_hat)

        self.log("test_loss", answer_loss)
        self.log("test_bleu", bleu_score)
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        scheduler_dict = self.lr_scheduler(optimizer)
        return (
            {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}
        )


class NameGenLogging(L.Callback):
    def __init__(self, beamsize:int = 5) -> None:
        super().__init__()
        self.male_name_start = "M"
        self.female_name_start = "F"
    
    def generation(self, generator:BeamGenerator, pl_module: Model_Lightning_Shell, male:bool = True):
        if male:
            start_name = pl_module.tokenizer.encode(self.male_name_start)
        else:
            start_name = pl_module.tokenizer.encode(self.female_name_start)
        
        gens = generator(
            seed_text = start_name,
            beamsize = pl_module.args.generation.beamsize,
            max_steps_n = pl_module.args.data.chunk_size,
            return_hypotheses_n = 2,
            need_reweight = pl_module.args.generation.val_reweight,
            temperature = pl_module.args.generation.temperature,
            alpha = pl_module.args.generation.alpha,
            without_score = True,
            need_to_encode = False
        )
        return gens


    def generation_and_log(self, pl_module: Model_Lightning_Shell):
        beam_gen = BeamGenerator(pl_module, pl_module.tokenizer)

        male_gens = self.generation(beam_gen, pl_module, male = True)
        female_gens = self.generation(beam_gen, pl_module, male = False)

        pl_module.logger.log_text(key = "Gen names", colums = ["male", "female"], data = [male_gens, female_gens])
    
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.generation_and_log(pl_module)
    
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.generation_and_log(pl_module)

class VowelsConsonantsAlternation_Metric(object):
    """Метрика для подсчёта, насколько читаемо получилось слово\n
    Метрика для подсчёта числового значения читаемости слова через соотношения переходов гласных и согласных.\n
    Формула: 
        `VCA = (1 / len(word)) * (sum(+)) / (sum(-) + 1)`,\n
        где `len(word)` - длина слова;\n
            `sum(+)` - количество переходов с согласной буквы на гласную или наоборот;\n
            `sum(-)` - количество переходов с согласной буквы на согласную (или с гласной на гласную)\n
    `VCA` примимает значение в диапазоне `[0, 1)`. Опытным путём установлено, что для читаемости слова значение метрики необходимо не менее `0.25`.
    """
    def __init__(self, mode:Literal["en"] = "en", tokenizer:CharTokenizer|None = None) -> None:
        self.mode = mode
        if mode == "en":
            self.vowels = [
                "a", "e", "i", "o", "u", "y",
            ]
            self.conconants = [
                "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "z"
            ]

        if tokenizer is not None:
            self.vowels = [tokenizer.encode(char) for char in self.vowels]
            self.conconants = [tokenizer.encode(char) for char in self.conconants]
            self.pad = tokenizer.pad_value
            self.mode = "tokens"

    def __call__(self, word_orig) -> float:
        word = [char for char in word_orig if char in self.vowels or char in self.conconants]

        sum_plus = 0
        sum_minus = 0

        first_word = True
        atlernation = -1

        for char in word:
            if first_word:
                if char in self.vowels:
                    atlernation = 0
                else:
                    atlernation = 1
                first_word = False
            else:
                if char in self.vowels:
                    if atlernation == 0:
                        sum_minus += 1
                    else:
                        sum_plus += 1
                    atlernation = 0
                else:
                    if atlernation == 1:
                        sum_minus += 1
                    else:
                        sum_plus += 1
                    atlernation = 1
        try:
            metric = (1 / len(word)) * (sum_plus) / (sum_minus + 1)
        except Exception as error:
            print(word_orig)
            print(word)
            print(error)


        return metric