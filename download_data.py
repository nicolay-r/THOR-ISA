import argparse
import os
from os.path import dirname, join, realpath

from src.service import CsvService, THoRFrameworkService, download


current_dir = dirname(realpath(__file__))
DATA_DIR = join(current_dir, "data")

DS_NAME = "se24"
DS_DIR = join(DATA_DIR, DS_NAME)


def convert_se24_prompt_dataset(src, target):
    records_it = [[item[0], item[1], int(item[2]), int(item[3])]
                  for item in CsvService.read(target=src, skip_header=True,
                                              cols=["prompt", "source", "label", "label"])]
    THoRFrameworkService.write_dataset(target_template=target, entries_it=records_it, label_map={1: 1, 0: 0},
                                       is_implicit=lambda origin_label: origin_label != 0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest="train_data", type=str)
    parser.add_argument('--valid', dest="valid_data", type=str)
    parser.add_argument('--test', dest="test_data", type=str)
    args = parser.parse_args()

    data = {
        join(DS_DIR, "train_en.csv"): args.train_data,
        join(DS_DIR, "valid_en.csv"): args.valid_data,
        join(DS_DIR, "final_en.csv"): args.test_data,
    }

    ds_name = DS_NAME[0].upper() + DS_NAME[1:]

    pickle_se2024_data = {
        join(DS_DIR, f"{ds_name}_train"): join(DS_DIR, "train_en.csv"),
        join(DS_DIR, f"{ds_name}_valid"): join(DS_DIR, "valid_en.csv"),
        join(DS_DIR, f"{ds_name}_test"): join(DS_DIR, "final_en.csv"),
    }

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for target, url in data.items():
        download(dest_file_path=target, source_url=url)

    for target, src in pickle_se2024_data.items():
        convert_se24_prompt_dataset(src, target)