import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import *
from transformers.models.llama.modeling_llama import LlamaForCausalLM


class LlamaAdaptor(nn.Module):
    """
    A class for adapting English Llama to other languages
    """

    def __init__(self, src_model, tgt_model):
        """
        src_model: (LLamaForCausalLM) of English
        tgt_model: (LLamaForCausalLM) of Foreign
        """
        super(LlamaAdaptor, self).__init__()
        self.src_model = src_model
        self.tgt_model = tgt_model

        # force sharing params
        self.tgt_model.model.layers = self.src_model.model.layers
        
    def save_pretrained(self, path, is_main_process=None, save_function=None):
        self.tgt_model.save_pretrained(path, is_main_process, save_function)

    def forward(self, lang, input_ids, attention_mask=None, labels=None):
        model = self.src_model if lang == 'en' else self.tgt_model
        return model(input_ids, labels=labels, attention_mask=attention_mask)
