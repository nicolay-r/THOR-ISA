import csv
import os
import pickle
import sys
from zipfile import ZipFile

import requests
from tqdm import tqdm


def download(dest_file_path, source_url):
    print(('Downloading from {src} to {dest}'.format(src=source_url, dest=dest_file_path)))

    sys.stdout.flush()
    datapath = os.path.dirname(dest_file_path)

    if not os.path.exists(datapath):
        os.makedirs(datapath, mode=0o755)

    dest_file_path = os.path.abspath(dest_file_path)

    r = requests.get(source_url, stream=True)
    total_length = int(r.headers.get('content-length', 0))

    with open(dest_file_path, 'wb') as f:
        pbar = tqdm(total=total_length, unit='B', unit_scale=True)
        for chunk in r.iter_content(chunk_size=32 * 1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)


class TxtService:

    @staticmethod
    def read_lines(filepath):
        print("Opening file: {}".format(filepath))
        with open(filepath, "r") as f:
            return [line.strip() for line in f.readlines()]


class THoRFrameworkService:

    @staticmethod
    def __write(target, content):
        print(f"Write: {target}")
        with open(target, 'wb') as f:
            pickle.dump(content, f)

    @staticmethod
    def write_dataset(target_template, entries_it, label_map, is_implicit):
        """ THoR-related service for sampling.
        """
        assert(isinstance(label_map, dict))

        records = []
        for e in entries_it:
            assert(isinstance(e, list) and len(e) == 4)
            assert(isinstance(e[0], str))   # Text
            assert(isinstance(e[1], str))   # Entity
            assert(isinstance(e[2], int))   # Explicit Label
            assert(isinstance(e[3], int))   # Implicit label
            e[2] = label_map[e[2]]
            e[3] = 1 if is_implicit(e[3]) else 0
            records.append(e)

        THoRFrameworkService.__write(target=f"{target_template}.pkl", content=records)


class CsvService:

    @staticmethod
    def read(target, delimiter='\t', quotechar='"', skip_header=False, cols=None, return_row_ids=False):
        assert(isinstance(cols, list) or cols is None)

        header = None
        with open(target, newline='\n') as f:
            for row_id, row in enumerate(csv.reader(f, delimiter=delimiter, quotechar=quotechar)):
                if skip_header and row_id == 0:
                    header = row
                    continue

                # Determine the content we wish to return.
                if cols is None:
                    content = row
                else:
                    row_d = {header[col_name]: value for col_name, value in enumerate(row)}
                    content = [row_d[col_name] for col_name in cols]

                # Optionally attach row_id to the content.
                yield [row_id] + content if return_row_ids else content


class RuSentNE2023CodalabService:

    @staticmethod
    def save_submission(target, labels):
        assert(isinstance(labels, list))
        for l in labels:
            assert (isinstance(l, int))

        with ZipFile(target, "w") as zip_file:
            results = "\n".join([str(l) for l in labels])
            zip_file.writestr(f'baseline_results.txt', results)
