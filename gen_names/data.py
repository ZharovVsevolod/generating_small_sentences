import torch
from torch.utils.data import Dataset, DataLoader, random_split
import lightning as L

from typing import Any, List, Tuple, Literal
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import os
import requests, zipfile, io

class CharTokenizer():
    def __init__(self, mode:Literal["en"] = "en") -> None:
        self.pad_value = 0

        if mode == "en":
            self.dictionary = {
                "<PAD>": 0,
                "<BOS>": 1,
                "<UNK>" : 2,
                "<EOS>" : 3,
                "a" : 4,
                "b" : 5,
                "c" : 6,
                "d" : 7,
                "e" : 8,
                "f" : 9,
                "g" : 10,
                "h" : 11,
                "i" : 12,
                "j" : 13,
                "k" : 14,
                "l" : 15,
                "m" : 16,
                "n" : 17,
                "o" : 18,
                "p" : 19,
                "q" : 20,
                "r" : 21,
                "s" : 22,
                "t" : 23,
                "u" : 24,
                "v" : 25,
                "w" : 26,
                "x" : 27,
                "y" : 28,
                "z" : 29,
                "M": 30, # for male name
                "F": 31 # for female name
            }

    def encode(self, text:str, bos:bool = False, eos:bool = False) -> List[int]:
        text_idx = []
        if bos:
            text_idx.append(self.dictionary["<BOS>"])
        for char in text:
            try:
                text_idx.append(self.dictionary[char])
            except:
                text_idx.append(self.dictionary["<UNK>"])
        if eos:
            text_idx.append(self.dictionary["<EOS>"])
        return text_idx

    def decode(self, text_idx:List[int]) -> str:
        text = ""
        for index in text_idx:
            if index not in [0, 1, 3]:
                text += list(self.dictionary.keys())[list(self.dictionary.values()).index(index)]
        return text

class NamesDataset(Dataset):
    def __init__(self, names_encoded:List[List[int]], chunk_lenght:int = 16, pad_value:int = 0) -> None:
        super().__init__()
        self.names = names_encoded
        self.chunk_lenght = chunk_lenght
        self.pad_value = pad_value
    
    def __len__(self) -> int:
        return len(self.names)
    
    def ensure_length(self, txt:str) -> str:
        if len(txt) < self.chunk_lenght:
            txt = list(txt) + [self.pad_value] * (self.chunk_lenght - len(txt))
        else:
            txt = txt[:self.chunk_lenght]
        return txt

    def __getitem__(self, index:int) -> Tuple[str, str]:
        name = self.names[index]

        seed_name = name[:-1]
        target_name = name[1:]

        seed_name = self.ensure_length(seed_name)
        target_name = self.ensure_length(target_name)

        seed_name = torch.tensor(seed_name)
        target_name = torch.tensor(target_name)

        return seed_name, target_name

class NamesDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir:str,
            batch_size:int,
            chunk_size:int,
            tokenizer: CharTokenizer | None = None
        ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = CharTokenizer()
    
    def download(self):
        print("Downloading dataset...")
        print("If it is calling an error, please visit this site and download names dataset directly:")
        print("https://catalog.data.gov/dataset/baby-names-from-social-security-card-applications-national-data")
        print("Yoy should extract it in path of `datasets/`")
        request = requests.get("https://www.ssa.gov/oact/babynames/names.zip")
        zip_file = zipfile.ZipFile(io.BytesIO(request.content))
        zip_file.extractall(self.data_dir)
        print("Dataset has been downloaded")

    def prepare_data(self) -> None:
        if not os.path.isdir(self.data_dir):
            self.download()
        else:
            print("Dataset is on his place")

    def load_and_encode_full_text(self):
        full_names = []
        files_in_folder = os.listdir(self.data_dir)

        for filename in files_in_folder:
            with open(self.data_dir + "/" + filename, "r", encoding="utf-8") as file:
                names = file.read().split('\n')

                names = [name.split(",")[:2] for name in names] # "Alice,F,26" -> ["Alice", "F"]
                names = [name[-1] + name[0].lower() for name in names] # ["Alice", "F", "26"] -> "Falice"
                names = [self.tokenizer.encode(name, bos=True, eos=True) for name in names] # "Falice" -> [1, 31, 4, 15, 12, 6, 8, 3]

                full_names += names

        return full_names
    
    def setup(self, stage: str) -> None:
        print("Loading and encoding all names dataset...")
        full_names = self.load_and_encode_full_text()
        print("Names had been loaded and encoded")
        print("Splitting the full dataset")
        full_names = full_names
        names_train, names_val = random_split(full_names, [0.8, 0.2])
        names_val, names_test = random_split(names_val, [0.5, 0.5])
        del full_names
        print("Dataset had been splited")

        if stage == "fit" or stage is None:
            self.train_dataset = NamesDataset(
                names_train, 
                chunk_lenght = self.chunk_size, 
                pad_value = self.tokenizer.pad_value
            )
            self.val_dataset = NamesDataset(
                names_val, 
                chunk_lenght = self.chunk_size, 
                pad_value = self.tokenizer.pad_value
            )
            print("Stage `fit` is set")

        if stage == "test" or stage is None:
            self.test_dataset = NamesDataset(
                names_test, 
                chunk_lenght = self.chunk_size, 
                pad_value = self.tokenizer.pad_value
            )
            print("Stage `test` is set")
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )