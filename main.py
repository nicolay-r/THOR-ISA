import argparse

import yaml
import torch
from attrdict import AttrDict
import pandas as pd

from download_data import DS_CAUSE_NAME, DS_CAUSE_S1_NAME, DS_STATE_NAME
from src.engine_prompt import PromptTrainer
from src.engine_thor_cause import ThorCauseTrainer
from src.engine_thor_cause_rr import ThorCauseReasoningRevisionTrainer
from src.engine_thor_state import ThorStateTrainer
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
        if config.infer_iter is True:
            config.shuffle = False
        self.config = config

    def forward(self):
        print(f"Loading data. Shuffle mode: {self.config.shuffle}")

        (self.trainLoader, self.validLoader, self.testLoader), self.config = MyDataLoader(self.config).get_data()

        self.model = LLMBackbone(config=self.config).to(self.config.device)
        self.config = load_params_LLM(self.config, self.model, self.trainLoader)
        print("Learning Rate (for training): ", self.config.bert_lr)

        print(f"Running on the {self.config.data_name} data.")
        if self.config.reasoning in ['prompt_state', 'prompt_cause']:
            print("Choosing prompt one-step infer mode.")
            print(f"Infer instruction: {self.config.instruct}")
            trainer = PromptTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'thor_state':
            print("Choosing thor multi-step THoR-State infer mode.")
            trainer = ThorStateTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'thor_cause':
            print("Choosing thor multi-step THoR-Cause infer mode.")
            trainer = ThorCauseTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'thor_cause_rr':
            print("Choosing thor multi-step THoR-Cause with Reasoning Revision infer mode.")
            trainer = ThorCauseReasoningRevisionTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        else:
            raise Exception('Should choose a correct reasoning mode: prompt or thor.')

        epoch_from = 0

        e_load = None
        if self.config.load_iter >= 0:
            assert(self.config.load_iter < self.config.epoch_size)
            e_load = self.config.load_iter if self.config.load_iter >= 0 else None
            print(f"Loading the pre-trained state: {e_load}")
            trainer.load_from_epoch(epoch=self.config.load_iter)
            epoch_from = e_load + 1
        if self.config.load_path is not None:
            print(f"Loading the pre-trained state: {self.config.load_path}")
            trainer.load_from_path(state_path=self.config.load_path)
        if self.config.validate is True:
            print("Zero-shot mode for evaluation.")
            r = trainer.evaluate_step(self.validLoader, 'valid')
            print(r)
            return
        if self.config.infer_iter is True:
            print(f"Final inference. Loading state: {e_load}. Shuffle mode: {self.config.shuffle}")
            r = trainer.final_infer(dataLoader=self.testLoader)

            if self.config.reasoning == 'thor_cause_rr':
                lines_it = [list(l) for l in zip(r["cause"]["total"], r["state"]["total"])]
                header = ["cause", "state"]
            else:
                lines_it = [[l] for l in r["total"]]
                header = ["cause"]

            submission_name = f"{self.config.model_path.replace('/', '_')}-{self.config.reasoning}-{e_load}.csv"
            CsvService.write(target=submission_name, lines_it=lines_it, header=header)
            return

        print("Fine-tuning mode for training.")
        trainer.train(epoch_from=epoch_from)
        lines = trainer.lines

        df = pd.DataFrame(lines)
        print(df.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda_index', default=0)
    parser.add_argument('-r', '--reasoning', default=None,
                        choices=['prompt_state', 'prompt_cause', 'thor_state', 'thor_cause', 'thor_cause_rr'],
                        help='with one-step prompt or multi-step thor reasoning')
    parser.add_argument('-v', '--validate', action='store_true', default=False,
                        help='running under zero-shot mode or fine-tune mode')
    parser.add_argument('-i', '--infer_iter', action='store_true', default=False,
                        help='running infer on specific index')
    parser.add_argument('-li', '--load_iter', default=-1, type=int, help='load a state on specific index')
    parser.add_argument('-lp', '--load_path', default=None, type=str, help="load a state on specific path")
    parser.add_argument('-d', '--data_name', default=None, choices=[DS_CAUSE_NAME, DS_STATE_NAME, DS_CAUSE_S1_NAME])
    parser.add_argument('-f', '--config', default='./config/config.yaml', help='config file')
    parser.add_argument('-p', '--instruct', default=None, type=str,
                        help="instructive prompt for `prompt` training engine that involves `target` parameter only")
    parser.add_argument('-dbg', '--debug', action='store_true', default=False)
    parser.add_argument('-es', '--epoch_size', default=None, type=int)
    parser.add_argument('-bs', '--batch_size', default=None, type=int)
    parser.add_argument('-lr', '--bert_lr', default=2e-4, type=float)

    default_instructs = {
        "prompt_cause": "What emotion causes '{target}' towards the last conversation utterance?",
        "prompt_state": "What emotion state is expressed in '{target}'?",
    }

    args = parser.parse_args()

    if args.instruct is None and args.reasoning in default_instructs:
        args.instruct = default_instructs[args.reasoning]

    template = Template(args)
    template.forward()
