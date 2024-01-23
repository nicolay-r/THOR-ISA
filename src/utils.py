import os
import random
import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


def prompt_direct_inferring_emotion(config, context, target):
    new_context = f"Given the conversation: {context}. "
    prompt = new_context + \
             config.instruct.format(target=target) + \
             " Choose from: {}.".format(", ".join(config.label_list))
    return prompt


def prompt_for_aspect_inferring(context, target):
    new_context = f'Given the conversation "{context}", '
    prompt = new_context + f'which specific text span of {target} is possibly causes emotion?'
    return new_context, prompt


def prompt_for_opinion_inferring(context, target, aspect_expr):
    new_context = context + ' The mentioned text span is about ' + aspect_expr + '.'
    prompt = new_context + f' Based on the common sense, ' \
                           f'what is the implicit opinion towards the cause of mentioned text span of {target}, and why?'
    return new_context, prompt


def prompt_for_polarity_inferring(context, target, opinion_expr):
    new_context = context + f' The opinion towards the text span of {target} that causes emotion is ' + opinion_expr + '.'
    prompt = new_context + f' Based on such opinion, what is the emotion caused by {target} towards the last conversation utterance?'
    return new_context, prompt


def prompt_for_polarity_label(context, polarity_expr, label_list):
    prompt = context + f' The emotion caused is {polarity_expr}.' + \
             " Based on these contexts, summarize and return the emotion cause only." + \
             " Choose from: {}.".format(", ".join(label_list))
    return prompt


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_params_LLM(config, model, fold_data):
    no_decay = ['bias', 'LayerNorm.weight']
    named = (list(model.named_parameters()))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named if not any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr),
         'weight_decay': float(config.weight_decay)},
        {'params': [p for n, p in named if any(nd in n for nd in no_decay)],
         'lr': float(config.bert_lr),
         'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=float(config.adam_epsilon))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=config.epoch_size * fold_data.__len__())
    config.score_manager = ScoreManager()
    config.optimizer = optimizer
    config.scheduler = scheduler
    return config


class ScoreManager:
    def __init__(self) -> None:
        self.score = []
        self.line = []

    def add_instance(self, score, res):
        self.score.append(score)
        self.line.append(res)

    def get_best(self):
        best_id = np.argmax(self.score)
        res = self.line[best_id]
        return res
