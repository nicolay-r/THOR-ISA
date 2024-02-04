import argparse
import os
from os.path import dirname, join, realpath

from src.service import CsvService, THoRFrameworkService, download


LABEL_MAP = {1: 1, 0: 0, -1: 2}
LABEL_MAP_REVERSE = {v: k for k, v in LABEL_MAP}


def convert_rusentne2023_dataset(src, target):
    print(f"Reading source: {src}")
    records_it = [[item[0], item[1], int(item[2])]
                  for item in CsvService.read(target=src, skip_header=True, cols=["sentence", "entity", "label"])]
    THoRFrameworkService.write_dataset(target_template=target, entries_it=records_it, label_map=LABEL_MAP)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--valid', dest="valid_data", type=str)
    parser.add_argument('--test', dest="test_data", type=str)
    args = parser.parse_args()

    current_dir = dirname(realpath(__file__))
    DATA_DIR = join(current_dir, "data")

    data = {
        # Data related to RuSentNE competitions.
        join(DATA_DIR, "rusentne2023/train_en.csv"): "https://www.dropbox.com/scl/fi/szj5j87f6w3ershnfh39x/train_data_en.csv?rlkey=h6ve617kl3o8g57otbt3yzamv&dl=1",
        join(DATA_DIR, "rusentne2023/valid_en.csv"): args.valid_data,
        join(DATA_DIR, "rusentne2023/final_en.csv"): args.test_data,
    }

    pickle_rusentne2023_data = {
        join(DATA_DIR, "rusentne2023/Rusentne2023_train"): join(DATA_DIR, "rusentne2023/train_en.csv"),
        join(DATA_DIR, "rusentne2023/Rusentne2023_valid"): join(DATA_DIR, "rusentne2023/valid_en.csv"),
        join(DATA_DIR, "rusentne2023/Rusentne2023_test"): join(DATA_DIR, "rusentne2023/final_en.csv"),
    }

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for target, url in data.items():
        download(dest_file_path=target, source_url=url)

    for target, src in pickle_rusentne2023_data.items():
        convert_rusentne2023_dataset(src, target)
