import os
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict, Counter
from src.cot_cause import ChainOfThoughtCause


class ThorCauseTrainer:
    """ This is a modified version of the trainer that involves Reasoning Revision.
    """

    def __init__(self, model, config, train_loader, valid_loader, test_loader) -> None:
        self.model = model
        self.config = config
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        self.save_name = os.path.join(config.target_dir, config.save_name)
        self.final_score = 0
        self.final_res = ''
        self.scores, self.lines = [], []
        self.re_init()

    def train(self, epoch_from):
        best_score, best_iter = 0, -1
        for epoch in tqdm(range(self.config.epoch_size)):
            if epoch < epoch_from:
                continue
            self.model.global_epoch = epoch
            self.global_epoch = epoch
            self.train_step()
            result = self.evaluate_step(mode='valid')
            self.re_init()
            score = result['default']
            self.add_instance(result)
            res = self.get_best()

            if score > best_score:
                best_score, best_iter = score, epoch
                save_name = self.save_name.format(epoch)
                if not os.path.exists(self.config.target_dir):
                    os.makedirs(self.config.target_dir)
                torch.save({'epoch': epoch, 'model': self.model.cpu().state_dict(), 'best_score': best_score},
                           save_name)
                self.model.to(self.config.device)
            elif epoch - best_iter > self.config.patience:
                print("Not upgrade for {} steps, early stopping...".format(self.config.patience))
                break
            self.model.to(self.config.device)

        res = self.final_evaluate(best_iter)
        score = res['default']
        self.add_instance(res)
        save_name = self.save_name.format(epoch)
        self.final_score, self.final_res = score, res

    def prepare_step_two(self, aspect_exprs, data):
        context_A_ids, target_ids = [data[w] for w in 'context_A_ids, target_ids'.strip().split(', ')]
        contexts_A = [self.model.tokenizer.decode(ids) for ids in context_A_ids]
        contexts_A = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_A]
        targets = [self.model.tokenizer.decode(ids) for ids in target_ids]
        targets = [context.replace('<pad>', '').replace('</s>', '').strip() for context in targets]

        new_prompts = []
        contexts_B = []
        for context, target, aspect_expr in zip(contexts_A, targets, aspect_exprs):
            context_B, prompt = ChainOfThoughtCause.prompt_for_opinion_inferring(context, target, aspect_expr)
            new_prompts.append(prompt)
            contexts_B.append(context_B)

        batch_inputs = self.model.tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                              max_length=self.config.max_length)
        batch_inputs = batch_inputs.data
        batch_contexts_B = self.model.tokenizer.batch_encode_plus(contexts_B, padding=True, return_tensors='pt',
                                                                  max_length=self.config.max_length)
        batch_contexts_B = batch_contexts_B.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'context_B_ids': batch_contexts_B['input_ids'],
            'target_ids': target_ids,
        }

        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    def prepare_step_three(self, opinion_exprs, data, prompt_func):
        context_B_ids, target_ids = [data[w] for w in 'context_B_ids, target_ids'.strip().split(', ')]
        contexts_B = [self.model.tokenizer.decode(ids) for ids in context_B_ids]
        contexts_B = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_B]
        targets = [self.model.tokenizer.decode(ids) for ids in target_ids]
        targets = [context.replace('<pad>', '').replace('</s>', '').strip() for context in targets]

        new_prompts = []
        contexts_C = []
        for context, target, opinion_expr in zip(contexts_B, targets, opinion_exprs):
            context_C, prompt = prompt_func(context, target, opinion_expr)
            new_prompts.append(prompt)
            contexts_C.append(context_C)

        batch_contexts_C = self.model.tokenizer.batch_encode_plus(contexts_C, padding=True, return_tensors='pt',
                                                                  max_length=self.config.max_length)
        batch_contexts_C = batch_contexts_C.data
        batch_inputs = self.model.tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                              max_length=self.config.max_length)
        batch_inputs = batch_inputs.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'context_C_ids': batch_contexts_C['input_ids'],
        }
        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    def prepare_step_label_cause(self, polarity_exprs, pre_cxt, data, label_type, prompt_func):
        output_ids, output_masks = [data[w] for w in f'output_{label_type}_ids, output_{label_type}_masks'.strip().split(', ')]

        context_C_ids = pre_cxt['context_C_ids']
        contexts_C = [self.model.tokenizer.decode(ids) for ids in context_C_ids]
        contexts_C = [context.replace('<pad>', '').replace('</s>', '').strip() for context in contexts_C]

        new_prompts = []
        for context_C, polarity_expr in zip(contexts_C, polarity_exprs):
            prompt = prompt_func(context_C, polarity_expr, self.config.label_list)
            new_prompts.append(prompt)

        batch_inputs = self.model.tokenizer.batch_encode_plus(new_prompts, padding=True, return_tensors='pt',
                                                              max_length=len(self.config.label_list))
        batch_inputs = batch_inputs.data

        res = {
            'input_ids': batch_inputs['input_ids'],
            'input_masks': batch_inputs['attention_mask'],
            'output_ids': output_ids,
            'output_masks': output_masks,
        }
        res = {k: v.to(self.config.device) for k, v in res.items()}
        return res

    def train_step(self):
        self.model.train()
        train_data = tqdm(self.train_loader, total=self.train_loader.data_length)

        losses_cause = []
        losses_state = []
        for i, data in enumerate(train_data):
            step_one_inferred_output = self.model.generate(**data)

            step_one_inferred_data = self.prepare_step_two(step_one_inferred_output, data)
            step_two_inferred_output = self.model.generate(**step_one_inferred_data)

            # Infer cause.
            step_two_inferred_data_cause = self.prepare_step_three(
                step_two_inferred_output, step_one_inferred_data, prompt_func=ChainOfThoughtCause.prompt_for_emotion_cause_inferring)
            step_three_inferred_output_cause = self.model.generate(**step_two_inferred_data_cause)

            step_label_data_cause = self.prepare_step_label_cause(step_three_inferred_output_cause,
                                                                  step_two_inferred_data_cause, data, label_type="cause",
                                                                  prompt_func=ChainOfThoughtCause.prompt_for_emotion_cause_label)
            loss_cause = self.model(**step_label_data_cause)
            losses_cause.append(loss_cause.item())
            loss_cause.backward()

            # Infer state.
            step_two_inferred_data_state = self.prepare_step_three(
                step_two_inferred_output, step_one_inferred_data, prompt_func=ChainOfThoughtCause.prompt_for_emotion_state_inferring)
            step_three_inferred_output_state = self.model.generate(**step_two_inferred_data_state)

            step_label_data_state = self.prepare_step_label_cause(step_three_inferred_output_state,
                                                                  step_two_inferred_data_state, data, label_type="state",
                                                                  prompt_func=ChainOfThoughtCause.prompt_for_emotion_state_label)
            loss_state = self.model(**step_label_data_state)
            losses_state.append(loss_state.item())
            loss_state.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            description = "Epoch {}, loss_cause:{:.4f}, loss_state:{:.4f}".format(self.global_epoch, np.mean(losses_cause), np.mean(losses_state))
            train_data.set_description(description)

            self.config.optimizer.step()
            self.config.scheduler.step()
            self.model.zero_grad()

    def evaluate_step(self, dataLoader=None, mode='valid'):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
            with torch.no_grad():
                step_one_inferred_output = self.model.generate(**data)

                step_one_inferred_data = self.prepare_step_two(step_one_inferred_output, data)
                step_two_inferred_output = self.model.generate(**step_one_inferred_data)

                step_two_inferred_data_cause = self.prepare_step_three(step_two_inferred_output, step_one_inferred_data,
                                                                       prompt_func=ChainOfThoughtCause.prompt_for_emotion_cause_inferring)
                step_three_inferred_output_cause = self.model.generate(**step_two_inferred_data_cause)

                step_label_data_cause = self.prepare_step_label_cause(step_three_inferred_output_cause, step_two_inferred_data_cause, data, label_type="cause",
                                                                      prompt_func=ChainOfThoughtCause.prompt_for_emotion_cause_label)
                output_cause = self.model.evaluate(**step_label_data_cause)
                self.add_output_cause(data, output_cause)

        result = self.report_score(mode=mode)
        return result

    def final_evaluate(self, epoch=0):
        PATH = self.save_name.format(epoch)
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])
        self.model.eval()
        res = self.evaluate_step(self.valid_loader, mode='valid')
        self.add_instance(res)
        return res

    def infer_step(self, dataLoader=None):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader
        dataiter = dataLoader
        result = defaultdict(list)
        for i, data in tqdm(enumerate(dataiter), total=dataLoader.data_length):
            with torch.no_grad():
                output = self.model.evaluate(**data)
                result["total"] += output
        return result

    def load_from_epoch(self, epoch=0):
        PATH = self.save_name.format(epoch)
        self.model.load_state_dict(torch.load(PATH, map_location=self.config.device)['model'])

    def load_from_path(self, state_path=None):
        self.model.load_state_dict(torch.load(state_path, map_location=self.config.device)['model'])

    def final_infer(self, dataLoader):
        self.model.eval()
        res = self.infer_step(self.valid_loader if dataLoader is None else dataLoader)
        self.add_instance(res)
        return res

    def add_instance(self, res):
        self.lines.append(res)

    def get_best(self):
        best_id = np.argmax([w['default'] for w in self.lines])
        res = self.lines[best_id]
        return res

    def re_init(self):
        self.preds, self.golds = defaultdict(list), defaultdict(list)
        self.keys = ['total']

    def add_output_cause(self, data, output):
        gold = data['input_labels_cause']
        for i, key in enumerate(self.keys):
            if i == 0:
                self.preds[key] += output
                self.golds[key] += gold.tolist()

    def report_score(self, mode='valid'):
        c = Counter()
        for l in self.preds['total']:
            c[l] += 1

        res = {}
        res['Acc'] = accuracy_score(self.golds['total'], self.preds['total'])
        res["F1"] = f1_score(self.golds['total'], self.preds['total'], average='macro', labels=list(range(len(self.config.label_list))))
        res['default'] = res['F1']
        res['mode'] = mode
        res['labels'] = c
        for k, v in res.items():
            if isinstance(v, float):
                res[k] = round(v * 100, 3)
        return res
