import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from gen_names.models.shell import VowelsConsonantsAlternation_Metric

def load_and_encode_full_text(data_dir):
    full_names = []
    files_in_folder = os.listdir(data_dir)

    for filename in files_in_folder:
        with open(data_dir + "/" + filename, "r", encoding="utf-8") as file:
            names = file.read().split('\n')

            names = [name.split(",")[0] for name in names] # "Alice,F,26" -> ["Alice"]
            names = [name.lower() for name in names] # ["Alice", "F", "26"] -> "alice"
            names = [name for name in names if name != ""]

            full_names += names

    return full_names

def test_names_vca():
    vca = VowelsConsonantsAlternation_Metric()

    names = ["Anna", "Alice", "Robert", "Michal", "Ghtifs", "Vsevolod", "Alex", "Guiuhsp", "Uisapoj", "Natasha", "Ohafsjd"]

    for name in names:
        print(name, vca(name.lower()))


def full_data_vca():
    full_names = load_and_encode_full_text("datasets/names")
    vca = VowelsConsonantsAlternation_Metric()

    metric = np.array([])

    for name in tqdm(full_names):
        metric = np.append(metric, vca(name))
    
    print(f"max = {np.max(metric)}")
    print(f"min = {np.min(metric)}")
    print(f"avg = {np.average(metric)}")

    plt.hist(metric)
    plt.xlabel("VCA")
    plt.ylabel("Count")

    save_path = "images"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    plt.savefig(os.path.join(save_path, "true_names_vca.png"))


if __name__ == "__main__":
    full_data_vca()