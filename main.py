import argparse
from os.path import join

import yaml
import torch
from attrdict import AttrDict
import pandas as pd

from src.service import RuSentNE2023CodalabService
from src.utils import set_seed, load_params_LLM
from src.loader import MyDataLoader
from src.model import LLMBackbone
from src.engine import PromptTrainer, ThorTrainer


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
        if config.eval_iter >= 0:
            config.shuffle = False
        self.config = config

    def forward(self):
        print(f"Loading data. Shuffle mode: {self.config.shuffle}")
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
            raise 'Should choose a correct reasoning mode: prompt or thor.'

        if self.config.zero_shot == True:
            print("Zero-shot mode for evaluation.")
            r = trainer.evaluate_step(self.testLoader, 'test')
            print(r)
            return
        if self.config.eval_iter >= 0:
            print(f"Final evaluation. Loading state: {self.config.eval_iter}")
            r = trainer.final_evaluate(self.config.eval_iter)
            print(r)
            submission_name = f"{self.config.model_path.replace('/', '_')}-{self.config.eval_iter}-test-submisssion.zip"
            RuSentNE2023CodalabService.save_submission(target=join(self.config.preprocessed_dir, submission_name),
                                                       labels=trainer.preds['total'])
            return

        print("Fine-tuning mode for training.")
        trainer.train()
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
    parser.add_argument('-e', '--eval_iter', default=-1, type=int, help='running evaluation on specific index')
    parser.add_argument('-d', '--data_name', default='rusentne2023')
    parser.add_argument('-f', '--config', default='./config/config.yaml', help='config file')
    args = parser.parse_args()
    template = Template(args)
    template.forward()
