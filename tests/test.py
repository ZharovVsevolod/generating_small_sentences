import os
import requests, zipfile, io
from gen_names.data import CharTokenizer, NamesDataModule

def check_and_get_files(data_dir):
    files_in_folder = os.listdir(data_dir)
    # print(files_in_folder)
    # print()

    filename = files_in_folder[0]

    full_names = []

    with open(data_dir + "/" + filename, "r", encoding="utf-8") as file:
        tokenizer = CharTokenizer()

        text = file.read().split('\n')

        for i in [0, 5, 10, 15]:
            names = text[i:i+5]
            print(names)
            names = [name.split(",") for name in names] # "Alice,F,26" -> ["Alice", "F", "26"]
            print(names)
            names = [name[1] + name[0].lower() for name in names] # ["Alice", "F", "26"] -> "Falice"
            print(names)
            names = [tokenizer.encode(name, bos=True, eos=True) for name in names] # "Falice" -> [1, 31, 4, 15, 12, 6, 8, 3]
            print(names)
            print("-----")
            full_names += names
    
    print(full_names)

def download_names(data_dir):
    print("Downloading dataset...")
    request = requests.get("https://www.ssa.gov/oact/babynames/names.zip")
    zip_file = zipfile.ZipFile(io.BytesIO(request.content))
    zip_file.extractall(data_dir)
    print("Dataset has been downloaded")

def tokenizer_check(name = "Alice"):
    tokenizer = CharTokenizer()

    name_idx = tokenizer.encode(name.lower())
    print(name, "->", name_idx)

    name_output = tokenizer.decode(name_idx)
    print(name_idx, "->", name_output)

def datamodule_check(data_dir):
    dm = NamesDataModule(
        data_dir=data_dir, 
        batch_size=4, 
        chunk_size=16, 
        tokenizer=CharTokenizer()
    )
    dm.prepare_data()
    dm.setup("fit")

if __name__ == "__main__":
    data_dir = "datasets/names"

    # if not os.path.isdir(data_dir):
    #     download_names(data_dir)
    # else:
    #     print("Dataset is here")

    # check_and_get_files(data_dir)
    # tokenizer_check("Alice")
    datamodule_check(data_dir)