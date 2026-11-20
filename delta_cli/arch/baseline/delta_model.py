"""Baseline bundle: Qwen3.5 0.8B text model through transformers, no remote code.

Bundle layout: this file, ``config.json``, ``tokenizer.json`` and one or more
``*.safetensors`` files with the released ``model.language_model.*`` keys (the
vision tower is ignored). Start your own architecture by editing this file.
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from delta_cli.arch.helpers import gather_logprobs, last_logits

PREFIX = 'model.language_model.'


class Model:
    max_context = 32768
    batch_tokens = 65536

    def __init__(self, bundle, device):
        root = Path(bundle)
        config = json.loads((root / 'config.json').read_text())
        text = Qwen3_5TextConfig(**config.get('text_config', config))
        dtype = torch.bfloat16 if device != 'cpu' else torch.float32
        with torch.device(device):
            model = Qwen3_5ForCausalLM(text).to(dtype)
        state = {}
        for path in sorted(root.glob('*.safetensors')):
            for key, value in load_file(str(path), device=device).items():
                if key.startswith(PREFIX):
                    state['model.' + key[len(PREFIX):]] = value
                elif key == 'lm_head.weight':
                    state[key] = value
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or set(missing) - ({'lm_head.weight'} if text.tie_word_embeddings else set()):
            raise ValueError('Checkpoint does not match config')
        if text.tie_word_embeddings:
            model.tie_weights()
        self.model = model.eval()
        # Byte-level BPE without Qwen's NFC normalizer is exactly invertible.
        self.tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
        self.tokenizer.normalizer = None
        self.vocab_size = text.vocab_size
        self.eos_id = self.bos_id = text.eos_token_id  # <|endoftext|> frames each document

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def _hidden(self, ids):
        return self.model.model(input_ids=ids, use_cache=False).last_hidden_state

    def token_logprobs(self, ids, targets):
        return gather_logprobs(self._hidden(ids), self.model.lm_head.weight, targets)

    def next_logits(self, ids):
        return last_logits(self._hidden(ids), self.model.lm_head.weight)
