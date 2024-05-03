import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class Phi3Backbone(nn.Module):
    def __init__(self, config):
        super(Phi3Backbone, self).__init__()
        self.config = config

        checkpoint_path = "microsoft/Phi-3-mini-4k-instruct"
        model_kwargs = dict(
            use_cache=False,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # loading the model with flash-attenstion support
            torch_dtype=torch.bfloat16,
            device_map=None
        )
        self.engine = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.tokenizer.model_max_length = 2048
        self.tokenizer.pad_token = self.tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.tokenizer.padding_side = 'right'

    def forward(self, **kwargs):
        input_ids, input_masks, output_ids, output_masks = [kwargs[w] for w in '\
        input_ids, input_masks, output_ids, output_masks'.strip().split(', ')]
        output_ids[output_ids[:, :] == self.tokenizer.pad_token_id] = -100
        output = self.engine(input_ids, attention_mask=input_masks, decoder_input_ids=None,
                             decoder_attention_mask=output_masks, labels=output_ids)
        loss = output[0]
        return loss

    def __map_label(self):
        # TODO. Setup label mapping.
        pass

    def generate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks,
                                      max_length=self.config.max_length, temperature=self.config.temperature)
        dec = [self.tokenizer.decode(ids) for ids in output]
        output = [context.replace('<pad>', '').replace('</s>', '').strip() for context in dec]
        return output

    def evaluate(self, **kwargs):
        input_ids, input_masks = [kwargs[w] for w in '\
        input_ids, input_masks'.strip().split(', ')]
        output = self.engine.generate(input_ids=input_ids, attention_mask=input_masks, max_length=200,
                                      temperature=self.config.temperature)
        dec = [self.tokenizer.decode(ids) for ids in output]
        label_dict = {w: i for i, w in enumerate(self.config.label_list)}
        output = [label_dict.get(w.lower().replace('<pad>', '').replace('</s>', '').strip(),
                                 label_dict[self.config.no_label]) for w in dec]
        return output
