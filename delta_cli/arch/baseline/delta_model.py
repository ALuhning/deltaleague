"""Baseline bundle: Qwen3.5 0.8B served by vLLM inside the sandbox (no network).

Bundle layout: this file, the model's ``config.json`` and tokenizer files, and
the released ``*.safetensors`` weights. vLLM scores whole windows with prompt
log-probabilities. ``VLLM_BATCH_INVARIANT`` makes every position's numbers
independent of batch composition and prefix length, so the audit's prefix-only
recomputation reproduces the scoring run exactly.

``delta_model_hf.py`` is the same model in plain PyTorch/transformers, a
starting point for architecture changes.
"""
import json
import os
from pathlib import Path

os.environ.setdefault('VLLM_BATCH_INVARIANT', '1')
os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
os.environ.setdefault('DO_NOT_TRACK', '1')

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


class Model:
    max_context = 32768
    gpu_memory_utilization = 0.70  # prompt log-probabilities need logits memory outside the KV budget

    def __init__(self, bundle, device):
        from vllm import LLM
        root = Path(bundle)
        config = json.loads((root / 'config.json').read_text())
        text = config.get('text_config', config)
        self.vocab_size = text['vocab_size']
        self.eos_id = self.bos_id = text['eos_token_id']  # <|endoftext|> frames each document
        # Byte-level BPE without Qwen's NFC normalizer is exactly invertible.
        self.tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
        self.tokenizer.normalizer = None
        self.llm = LLM(model=str(root), dtype='bfloat16', trust_remote_code=False, language_model_only=True,
                       skip_tokenizer_init=True, max_model_len=self.max_context + 2, enable_prefix_caching=False,
                       gpu_memory_utilization=self.gpu_memory_utilization, max_logprobs=-1, seed=0)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def score_windows(self, inputs, targets):
        """Unpadded windows -> per-target log-probabilities, one engine call for the whole batch."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        prompts = [TokensPrompt(prompt_token_ids=list(x) + [y[-1]]) for x, y in zip(inputs, targets)]
        params = SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False, temperature=0)
        out = []
        for result, ids in zip(self.llm.generate(prompts, params, use_tqdm=False), prompts):
            tokens = ids['prompt_token_ids']
            entries = result.prompt_logprobs
            out.append([entries[i][tokens[i]].logprob for i in range(1, len(tokens))])
        return out

    def token_logprobs(self, ids, targets):
        # Padded fallback; the runner prefers score_windows.
        rows = self.score_windows(ids.tolist(), targets.tolist())
        return torch.tensor(rows, dtype=torch.float32)

    def next_logits(self, ids):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        params = SamplingParams(max_tokens=1, logprobs=self.vocab_size, detokenize=False, temperature=0)
        result = self.llm.generate([TokensPrompt(prompt_token_ids=ids[0].tolist())], params, use_tqdm=False)[0]
        entries = result.outputs[0].logprobs[0]
        row = torch.full((self.vocab_size,), float('-inf'))
        for token, entry in entries.items():
            row[token] = entry.logprob
        return row[None]
