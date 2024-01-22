import argparse

import yaml
import torch
from attrdict import AttrDict
import pandas as pd

from download_data import DS_NAME
from src.engine_prompt import PromptTrainer
from src.engine_thor import ThorTrainer
from src.service import CsvService
from src.utils import set_seed, load_params_LLM
from src.loader import MyDataLoader
from src.model import LLMBackbone


class Template:
    def __init__(self, args):
        config = AttrDict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
        names = []
        for k, v in vars(args).items():
            setattr(config, k, v)
        config.dataname = config.data_name
        set_seed(config.seed)
        config.device = torch.device('cuda:{}'.format(config.cuda_index) if torch.cuda.is_available() else 'cpu')
        names = [config.model_size, config.dataname] + names
        config.save_name = '_'.join(list(map(str, names))) + '_{}.pth.tar'
        if config.infer_iter >= -1:
            config.shuffle = False
        self.config = config

    def forward(self):
        print(f"Loading data. Shuffle mode: {self.config.shuffle}")
        print(f"Loading data_prefix.{self.config.data_prefix}")

        (self.trainLoader, self.validLoader, self.testLoader), self.config = MyDataLoader(self.config).get_data()

        self.model = LLMBackbone(config=self.config).to(self.config.device)
        self.config = load_params_LLM(self.config, self.model, self.trainLoader)

        print(f"Running on the {self.config.data_name} data.")
        if self.config.reasoning == 'prompt':
            print("Choosing prompt one-step infer mode.")
            trainer = PromptTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'thor':
            print("Choosing thor multi-step infer mode.")
            trainer = ThorTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        else:
            raise Exception('Should choose a correct reasoning mode: prompt or thor.')

        epoch_from = 0

        if self.config.load_iter >= 0:
            e_load = self.config.load_iter if self.config.load_iter >= 0 else None
            print(f"Loading the pre-trained state: {e_load}")
            trainer.load(self.config.load_iter)
            epoch_from = e_load + 1
        if self.config.zero_shot == True:
            print("Zero-shot mode for evaluation.")
            r = trainer.evaluate_step(self.validLoader, 'valid')
            print(r)
            return
        if self.config.infer_iter >= -1:
            e_load = self.config.infer_iter if self.config.infer_iter >= 0 else None
            print(f"Final inference. Loading state: {e_load}. Shuffle mode: {self.config.shuffle}")
            r = trainer.final_infer(dataLoader=self.testLoader, epoch=e_load)
            submission_name = f"{self.config.model_path.replace('/', '_')}-{e_load}.csv"
            CsvService.write(target=submission_name, lines_it=[[l] for l in r["total"]], header=["label"])
            return

        print("Fine-tuning mode for training.")
        trainer.train(epoch_from=epoch_from)
        lines = trainer.lines

        df = pd.DataFrame(lines)
        print(df.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda_index', default=0)
    parser.add_argument('-r', '--reasoning', default='thor', choices=['prompt', 'thor'],
                        help='with one-step prompt or multi-step thor reasoning')
    parser.add_argument('-z', '--zero_shot', action='store_true', default=False,
                        help='running under zero-shot mode or fine-tune mode')
    parser.add_argument('-i', '--infer_iter', default=-2, type=int, help='running infer on specific index')
    parser.add_argument('-l', '--load_iter', default=-1, type=int, help='load a state on specific index')
    parser.add_argument('-d', '--data_name', default=DS_NAME)
    parser.add_argument('-d', '--data_prefix', default="cause", choices=["cause", "state"])
    parser.add_argument('-f', '--config', default='./config/config.yaml', help='config file')
    args = parser.parse_args()
    template = Template(args)
    template.forward()
