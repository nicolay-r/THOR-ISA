import argparse
from os.path import join

import yaml
import torch
from attrdict import AttrDict
import pandas as pd
from transformers import GenerationConfig

from src.cot_default import ChainOfThoughtDefault
from src.cot_v2 import ChainOfThoughtRuSentNE2023
from src.engine_prompt import PromptTrainer
from src.engine_thor import ThorTrainer
from src.service import RuSentNE2023CodalabService, CsvService
from src.utils import set_seed, load_params_LLM, OutputHandler
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
        if config.eval_iter >= 0:
            config.shuffle = False
        self.config = config

        # Setup COT mode.
        cot_choices = {
            "default": ChainOfThoughtDefault,
            "v2": ChainOfThoughtRuSentNE2023,
        }
        self.thor_cot = cot_choices[self.config.cot_mode]

    def forward(self):
        print(f"Loading data. Shuffle mode: {self.config.shuffle}")

        (self.trainLoader, self.validLoader, self.testLoader), self.config = \
            MyDataLoader(config=self.config, thor_cot=self.thor_cot).get_data()

        self.model = LLMBackbone(config=self.config).to(self.config.device)
        self.config = load_params_LLM(self.config, self.model, self.trainLoader)

        print(f"Running on the {self.config.data_name} data.")
        if self.config.reasoning == 'prompt':
            print("Choosing prompt one-step infer mode.")
            trainer = PromptTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader)
        elif self.config.reasoning == 'thor':
            print(f"Choosing THoR multi-step infer mode. [{type(self.thor_cot)}]")
            trainer = ThorTrainer(self.model, self.config, self.trainLoader, self.validLoader, self.testLoader,
                                  cot=self.thor_cot)
        else:
            raise Exception('Should choose a correct reasoning mode: prompt or thor.')

        if self.config.zero_shot == True:
            print("Zero-shot mode for evaluation.")
            r = trainer.evaluate_step(self.testLoader, 'test')
            print(r)
            return
        if self.config.eval_iter >= 0:
            print(f"Final evaluation. Loading state: {self.config.eval_iter}")
            h = OutputHandler()
            if self.config.reasoning == 'thor':
                trainer.output_handler = lambda text: h.forward(text)
            r = trainer.final_evaluate(self.config.eval_iter)
            print(r)
            submission_name = f"{self.config.model_path.replace('/', '_')}-{self.config.eval_iter}-test-submission.zip"
            label_map = {1: 1, 0: 0, 2: -1}
            RuSentNE2023CodalabService.save_submission(target=join(self.config.preprocessed_dir, submission_name),
                                                       labels=[label_map[l] for l in trainer.preds['total']])

            CsvService.write(lines_it=h.iter_chunks(3),
                             target=join(self.config.preprocessed_dir, submission_name + '.gen.csv'),
                             header=["s1_aspect", "s2_opinion", "s3_polarity"])
            return

        print("Fine-tuning mode for training.")
        trainer.train()
        lines = trainer.lines

        df = pd.DataFrame(lines)
        print(df.to_string())


if __name__ == '__main__':

    gen_config = GenerationConfig()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda_index', default=0)
    parser.add_argument('-r', '--reasoning', default='thor', choices=['prompt', 'thor'],
                        help='with one-step prompt or multi-step thor reasoning')
    parser.add_argument('-z', '--zero_shot', action='store_true', default=False,
                        help='running under zero-shot mode or fine-tune mode')
    parser.add_argument('-e', '--eval_iter', default=-1, type=int, help='running evaluation on specific index')
    parser.add_argument('-d', '--data_name', default='rusentne2023')
    parser.add_argument('-f', '--config', default='./config/config.yaml', help='config file')
    parser.add_argument('-t', '--temperature', default=gen_config.temperature, type=float,
                        help="Necessary for zero-shot option. For the training the default value of the "
                             "configuration from the `transformers` is better since we wish to get the same"
                             "result independing of the chosen path during generation.")
    parser.add_argument('-cm', '--cot_mode', default='default',
                        help="This is a Chain-of-Thought preset name parameter, necessary for "
                             "chosing the chains for the task.")

    args = parser.parse_args()
    template = Template(args)
    template.forward()
