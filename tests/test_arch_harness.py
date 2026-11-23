import math
import textwrap

import pytest

from delta_cli.arch.harness import ArchEvalError, Limits, evaluate
from delta_cli.arch.sandbox import docker_command, local_launcher

# Byte-level model: 256 bytes + bos + eos. Logits come from a fixed seeded
# embedding and projection, so every runner computes identical numbers.
HONEST = '''
import time
import torch

class Model:
    vocab_size, bos_id, eos_id, max_context = 258, 256, 257, 64

    def __init__(self, bundle, device):
        g = torch.Generator().manual_seed(7)
        self.emb = torch.randn(258, 16, generator=g)
        self.out = torch.randn(16, 258, generator=g)

    def encode(self, text):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode()

    def logits(self, ids):
        return self.emb[ids] @ self.out

    def token_logprobs(self, ids, targets):
        return self.logits(ids).log_softmax(-1).gather(-1, targets[..., None])[..., 0]

    def next_logits(self, ids):
        return self.logits(ids)[:, -1]
'''

CHEATS = {
    # Puts all mass on the true target, which it can see in phase A.
    'peek': 'def token_logprobs(self, ids, targets):\n    return torch.zeros(targets.shape)',
    # Unnormalized: skips log_softmax.
    'unnormalized': 'def token_logprobs(self, ids, targets):\n'
                    '    return (self.logits(ids).gather(-1, targets[..., None])[..., 0]).clamp(max=0)',
    'bad_tokens': 'def encode(self, text):\n    return [999] + list(text.encode())',
    'lossy': 'def encode(self, text):\n    return list(text.encode("ascii", "ignore"))\n'
             'def decode(self, ids):\n    return bytes(ids).decode()',
    'slow': 'def token_logprobs(self, ids, targets):\n    time.sleep(3)\n'
            '    return self.logits(ids).log_softmax(-1).gather(-1, targets[..., None])[..., 0]',
    'crash': 'def token_logprobs(self, ids, targets):\n    raise RuntimeError("secret text")',
    'killed': 'def token_logprobs(self, ids, targets):\n    import os, signal\n    os.kill(os.getpid(), signal.SIGKILL)',
    'runtime': 'def token_logprobs(self, ids, targets):\n    import os\n    os._exit(125)',
    'positive': 'def token_logprobs(self, ids, targets):\n    return torch.ones(targets.shape)',
    'chatty': 'def encode(self, text):\n    print("noise on stdout")\n    return list(text.encode())',
}
# Within tolerance per token, but a uniform positive bias.
BIASED = ('def token_logprobs(self, ids, targets):\n'
          '    return (self.logits(ids).log_softmax(-1).gather(-1, targets[..., None])[..., 0] + 0.05).clamp(max=0)')

DOCS = [{'text': 'Hello, delta league. ' * 6}, {'text': 'naïve café — ünïcode ✓'}, {'text': ''},
        {'text': 'truncated tail', 'predicts_eos': False}]


def bundle(tmp_path, name='honest', override=None):
    root = tmp_path / name
    root.mkdir()
    source = HONEST
    if override:
        source += '\n' + textwrap.indent(override, '    ') + '\n'
    (root / 'delta_model.py').write_text(source)
    return root


def run(root, limits=Limits(audit_count=200), docs=DOCS, seed=1):
    return evaluate(local_launcher(root), {'validation': docs}, limits, seed=seed)


def reference_nll(docs):
    import torch
    g = torch.Generator().manual_seed(7)
    emb, out = torch.randn(258, 16, generator=g), torch.randn(16, 258, generator=g)
    total = 0.0
    for doc in docs:
        targets = list(doc['text'].encode()) + ([257] if doc.get('predicts_eos', True) else [])
        for i in range(0, len(targets), 64):
            window = targets[i:i + 64]
            inputs = torch.tensor([256] + window[:-1])
            lp = (emb[inputs] @ out).log_softmax(-1)
            total -= float(lp[range(len(window)), window].double().sum())
    return total


def test_honest_model_scores_exactly(tmp_path):
    result = run(bundle(tmp_path))
    split = result['splits']['validation']
    assert split['nll_sum_nats'] == pytest.approx(reference_nll(DOCS), rel=1e-5)
    nbytes = sum(len(d['text'].encode()) for d in DOCS)
    assert split['utf8_bytes'] == nbytes
    assert split['bits_per_byte'] == pytest.approx(reference_nll(DOCS) / math.log(2) / nbytes, rel=1e-5)
    assert split['windows'] == 4 + 1  # 127-byte doc spans two 64-token windows
    audit = result['audit']
    assert audit['count'] == split['targets'] == 173 and audit['failures'] == 0 and audit['max_abs_deviation'] < 1e-4
    assert result['flops_per_token'] > 0 and result['contract']['vocab_size'] == 258


@pytest.mark.parametrize('cheat,code', [
    ('peek', 'audit_failed'), ('unnormalized', 'audit_failed'), ('bad_tokens', 'invalid_tokens'),
    ('lossy', 'roundtrip_failed'), ('crash', 'model_error'), ('positive', 'invalid_claims')])
def test_cheats_are_rejected(tmp_path, cheat, code):
    with pytest.raises(ArchEvalError) as exc:
        run(bundle(tmp_path, cheat, CHEATS[cheat]))
    assert exc.value.code == code
    assert 'secret' not in exc.value.detail


def test_killed_runner_reports_exit_status(tmp_path):
    with pytest.raises(ArchEvalError) as exc:
        run(bundle(tmp_path, 'killed', CHEATS['killed']))
    assert exc.value.code == 'model_error' and exc.value.detail == 'ProtocolError rc=-9'


def test_container_runtime_failure_is_not_a_model_error(tmp_path):
    from delta_cli.arch.harness import SandboxFailure
    with pytest.raises(SandboxFailure):
        run(bundle(tmp_path, 'runtime', CHEATS['runtime']))


def test_time_limit_kills_runner(tmp_path):
    with pytest.raises(ArchEvalError) as exc:
        run(bundle(tmp_path, 'slow', CHEATS['slow']), Limits(time_limit_seconds=1, audit_count=10))
    assert exc.value.code == 'time_limit_exceeded'


def test_flops_cap(tmp_path):
    root = bundle(tmp_path)
    flops = run(root)['flops_per_token']
    assert run(root, Limits(audit_count=10, flops_per_token_cap=flops * 1.2))
    with pytest.raises(ArchEvalError) as exc:
        run(root, Limits(audit_count=10, flops_per_token_cap=flops * 0.5))
    assert exc.value.code == 'flops_exceeded' and exc.value.stats['flops_per_token'] == flops


def test_uniform_bias_is_charged_back(tmp_path):
    honest = run(bundle(tmp_path))['splits']['validation']['nll_sum_nats']
    result = run(bundle(tmp_path, 'biased', BIASED))
    assert result['audit']['failures'] == 0 and result['audit']['penalty_nats_per_token'] > 0.04
    assert result['splits']['validation']['nll_sum_nats'] >= honest - 1e-3


def test_stdout_noise_does_not_break_protocol(tmp_path):
    assert run(bundle(tmp_path, 'chatty', CHEATS['chatty']))['audit']['failures'] == 0


def test_gpt2_normalized_loss_and_multiple_splits(tmp_path):
    docs = [{'text': 'abc', 'gpt2_targets': 2}, {'text': 'de', 'gpt2_targets': 2}]
    result = evaluate(local_launcher(bundle(tmp_path)), {'validation': docs, 'test': DOCS}, Limits(audit_count=50),
                      seed=3)
    val = result['splits']['validation']
    assert val['loss_per_gpt2_token'] == pytest.approx(val['nll_sum_nats'] / 4)
    assert 'loss_per_gpt2_token' not in result['splits']['test']


def test_audit_never_shows_a_target_before_it_is_answered():
    from delta_cli.arch.harness import Split, _audit_plan
    import random
    from array import array
    record = Split('v', [{'text': ''}] * 3)
    record.windows = [(0, array('i', range(50))), (0, array('i', range(50))), (1, array('i', range(30))),
                      (2, array('i', range(5)))]
    rounds = _audit_plan([record], 60, random.Random(0))
    seen = {}
    for layer in rounds:
        docs = [p[1] for p in layer]
        assert len(docs) == len(set(docs))  # one prefix per document per round
        for _, doc, w, p in layer:
            assert (w, p) > seen.get(doc, (-1, -1))
            seen[doc] = (w, p)
    assert sum(len(layer) for layer in rounds) == 60


def test_docker_command_is_locked_down(tmp_path):
    cmd = docker_command('registry/arch@sha256:' + 'a' * 64, tmp_path, 'n', runtime='runsc')
    joined = ' '.join(cmd)
    for flag in ('--network none', '--read-only', '--cap-drop ALL', 'no-new-privileges', '--user 65534:65534',
                 'readonly', '--runtime runsc', '/tmp:rw,exec,'):
        assert flag in joined
    with pytest.raises(ValueError):
        docker_command('registry/arch:latest', tmp_path, 'n')


def test_qwen_baseline_bundle_runs_and_audits(tmp_path):
    pytest.importorskip('transformers')
    import glob, json, shutil
    import torch
    from safetensors.torch import save_file
    from importlib.resources import files
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    tokenizers = glob.glob(str(__import__('pathlib').Path.home() /
                               '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/tokenizer.json'))
    if not tokenizers:
        pytest.skip('Qwen3.5 tokenizer not cached')
    root = tmp_path / 'qwen'
    root.mkdir()
    shutil.copy(files('delta_cli.arch.baseline') / 'delta_model_hf.py', root / 'delta_model.py')
    shutil.copy(tokenizers[0], root / 'tokenizer.json')
    text = dict(vocab_size=248320, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_value_heads=2,
                linear_num_key_heads=1, linear_key_head_dim=16, linear_value_head_dim=16,
                layer_types=['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'],
                tie_word_embeddings=True, eos_token_id=248044)
    (root / 'config.json').write_text(json.dumps({'model_type': 'qwen3_5', 'text_config': text}))
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**text))
    save_file({'model.language_model.' + k[len('model.'):]: v.contiguous()
               for k, v in model.state_dict().items() if k.startswith('model.')}, str(root / 'model.safetensors'))
    result = evaluate(local_launcher(root), {'validation': DOCS}, Limits(audit_count=40), seed=5)
    assert result['audit']['failures'] == 0 and result['audit']['max_abs_deviation'] < 1e-3
    # A random model is near-uniform over the vocabulary.
    split = result['splits']['validation']
    assert split['nats_per_token'] == pytest.approx(math.log(248320), rel=0.05)


def test_unpadded_score_windows_path(tmp_path):
    # Engines that batch internally (the vLLM baseline) implement score_windows.
    override = ('def score_windows(self, inputs, targets):\n'
                '    out = []\n'
                '    for x, y in zip(inputs, targets):\n'
                '        lp = self.logits(torch.tensor(x)).log_softmax(-1)\n'
                '        out.append(lp[range(len(y)), y].tolist())\n'
                '    return out')
    result = run(bundle(tmp_path, 'windows', override))
    assert result['splits']['validation']['nll_sum_nats'] == pytest.approx(reference_nll(DOCS), rel=1e-5)
    assert result['audit']['max_abs_deviation'] < 1e-4
