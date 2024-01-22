import argparse
import os
from os.path import dirname, join, realpath

import yaml
from attrdict import AttrDict

from src.service import CsvService, THoRFrameworkService, download


current_dir = dirname(realpath(__file__))
DATA_DIR = join(current_dir, "data")

DS_CAUSE_NAME = "cause-se24"
DS_CAUSE_DIR = join(DATA_DIR, DS_CAUSE_NAME)

DS_STATE_NAME = "state-se24"
DS_STATE_DIR = join(DATA_DIR, DS_STATE_NAME)


def convert_se24_prompt_dataset(src, target):
    records_it = [[item[0], item[1], int(item[2]), int(item[3])]
                  for item in CsvService.read(target=src, skip_header=True,
                                              cols=["prompt", "source", "label", "label"])]
    no_label_uint = config.label_list.index(config.no_label)
    print(f"No label: {no_label_uint}")
    THoRFrameworkService.write_dataset(target_template=target, entries_it=records_it,
                                       is_implicit=lambda origin_label: origin_label != no_label_uint)


def states_convert_se24_prompt_dataset(src, target):
    records_it = [[item[0], item[1], int(config.label_list.index(item[2])), int(config.label_list.index(item[2]))]
                  for item in CsvService.read(target=src, skip_header=True,
                                              cols=["prompt", "target", "emotion"])]
    no_label_uint = config.label_list.index(config.no_label)
    print(f"No label: {no_label_uint}")
    THoRFrameworkService.write_dataset(target_template=target, entries_it=records_it,
                                       is_implicit=lambda origin_label: origin_label != no_label_uint)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cause-train', dest="cause_train_data", type=str)
    parser.add_argument('--cause-valid', dest="cause_valid_data", type=str)
    parser.add_argument('--cause-test', dest="cause_test_data", type=str)
    parser.add_argument('--state-train', dest="state_train_data", type=str)
    parser.add_argument('--state-valid', dest="state_valid_data", type=str)
    parser.add_argument('--config', default='./config/config.yaml', help='config file')
    args = parser.parse_args()

    # Reading configuration.
    config = AttrDict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
    names = []
    for k, v in vars(args).items():
        setattr(config, k, v)

    data_sources = {
        join(DS_CAUSE_DIR, "cause_train_en.csv"): args.cause_train_data,
        join(DS_CAUSE_DIR, "cause_valid_en.csv"): args.cause_valid_data,
        join(DS_CAUSE_DIR, "cause_final_en.csv"): args.cause_test_data,
        join(DS_STATE_DIR, "state_train_en.csv"): args.state_train_data,
        join(DS_STATE_DIR, "state_valid_en.csv"): args.state_valid_data,
    }

    pickle_cause_se2024_data = {
        join(DS_CAUSE_DIR, f"{DS_CAUSE_NAME}_train"): join(DS_CAUSE_DIR, "cause_train_en.csv"),
        join(DS_CAUSE_DIR, f"{DS_CAUSE_NAME}_valid"): join(DS_CAUSE_DIR, "cause_valid_en.csv"),
        join(DS_CAUSE_DIR, f"{DS_CAUSE_NAME}_test"): join(DS_CAUSE_DIR, "cause_final_en.csv"),
    }

    pickle_state_se2024_data = {
        join(DS_STATE_DIR, f"{DS_STATE_NAME.capitalize()}_train"): join(DS_STATE_DIR, "state_train_en.csv"),
        join(DS_STATE_DIR, f"{DS_STATE_NAME.capitalize()}_valid"): join(DS_STATE_DIR, "state_valid_en.csv"),
        join(DS_STATE_DIR, f"{DS_STATE_NAME.capitalize()}_test"): join(DS_STATE_DIR, "state_valid_en.csv"),
    }

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for target, url in data_sources.items():
        download(dest_file_path=target, source_url=url)

    for target, src in pickle_cause_se2024_data.items():
        convert_se24_prompt_dataset(src, target)

    for target, src in pickle_state_se2024_data.items():
        states_convert_se24_prompt_dataset(src, target)
