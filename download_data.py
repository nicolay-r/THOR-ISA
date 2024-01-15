import os
from os.path import dirname, join, realpath

from src.service import CsvService, THoRFrameworkService, TxtService, download


def convert_rusentne2023_dataset(src, target):
    records_it = [[item[0], item[1], int(item[2]), int(item[3])]
                  for item in CsvService.read(target=src, skip_header=True, cols=["sentence", "entity", "label", "label"])]
    THoRFrameworkService.write_dataset(target_template=target, entries_it=records_it)


if __name__ == "__main__":
    current_dir = dirname(realpath(__file__))
    DATA_DIR = join(current_dir, "data")

    data = {
        # Data related to RuSentNE competitions.
        join(DATA_DIR, "rusentne2023/train_en.csv"): "https://www.dropbox.com/scl/fi/szj5j87f6w3ershnfh39x/train_data_en.csv?rlkey=h6ve617kl3o8g57otbt3yzamv&dl=1",
        join(DATA_DIR, "rusentne2023/valid_en.csv"): TxtService.read_lines(join(DATA_DIR, "rusentne2023_valid_data_link.txt"))[-1],
    }

    pickle_rusentne2023_data = {
        join(DATA_DIR, "rusentne2023/Rusentne2023_train"): join(DATA_DIR, "rusentne2023/train_en.csv"),
        join(DATA_DIR, "rusentne2023/Rusentne2023_valid"): join(DATA_DIR, "rusentne2023/valid_en.csv"),
        join(DATA_DIR, "rusentne2023/Rusentne2023_test"): join(DATA_DIR, "rusentne2023/train_en.csv"),
    }

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for target, url in data.items():
        download(dest_file_path=target, source_url=url)

    for target, src in pickle_rusentne2023_data.items():
        convert_rusentne2023_dataset(src, target)
