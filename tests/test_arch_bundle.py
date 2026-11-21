import json

import pytest
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from delta_cli.artifacts import validate_manifest, bundle_hash
from delta_cli.arch.bundle import build_manifest, scan
from delta_cli.main import app
from test_arch_harness import HONEST


@pytest.fixture
def honest(tmp_path):
    root = tmp_path / 'bundle'
    root.mkdir()
    (root / 'delta_model.py').write_text(HONEST)
    save_file({'w': torch.ones(2)}, str(root / 'extra.safetensors'))
    return root


def test_build_manifest_and_identity(honest):
    manifest = build_manifest(honest, '1b-arch')
    assert json.loads((honest / 'manifest.json').read_text()) == manifest
    validate_manifest(manifest)
    assert [a['name'] for a in manifest['artifacts']] == ['delta_model.py', 'extra.safetensors']
    # Rebuilding ignores the manifest itself; identity is the inventory.
    assert bundle_hash(build_manifest(honest, '1b-arch')['artifacts']) == bundle_hash(manifest['artifacts'])
    (honest / 'delta_model.py').write_text(HONEST + '\n# tweak\n')
    assert bundle_hash(scan(honest)) != bundle_hash(manifest['artifacts'])


@pytest.mark.parametrize('name', ['weights.bin', 'model.pt', 'evil.pkl', 'run.sh'])
def test_unsafe_files_are_rejected(honest, name):
    (honest / name).write_bytes(b'x')
    with pytest.raises(ValueError):
        build_manifest(honest, 'm')


def test_manifest_rules():
    base = {'schema_version': 'delta.submission.v2', 'format': 'bundle', 'model_config_id': 'm',
            'artifacts': [{'name': 'delta_model.py', 'sha256': 'a' * 64, 'size_bytes': 10}]}
    validate_manifest(base)
    for bad in ({'artifacts': [{'name': 'other.py', 'sha256': 'a' * 64, 'size_bytes': 1}]},
                {'artifacts': base['artifacts'] + [{'name': '../x.py', 'sha256': 'a' * 64, 'size_bytes': 1}]},
                {'artifacts': base['artifacts'] * 2},
                {'artifacts': [{'name': 'delta_model.py', 'sha256': 'a' * 64, 'size_bytes': 2 * 1024**2}]},
                {'parent_version_id': 'nope'}, {'format': 'full'}):
        with pytest.raises(ValueError):
            validate_manifest({**base, **bad})


def test_non_finite_weights_rejected(honest):
    save_file({'w': torch.tensor([float('nan')])}, str(honest / 'extra.safetensors'))
    with pytest.raises(ValueError, match='non-finite'):
        build_manifest(honest, 'm')


def test_cli_check_and_build(honest, tmp_path):
    docs = tmp_path / 'docs.jsonl'
    docs.write_text('\n'.join(json.dumps({'text': t}) for t in ['hello world', 'second doc ✓']) + '\n')
    runner = CliRunner()
    result = runner.invoke(app, ['bundle', 'check', str(honest), '--docs', str(docs), '--device', 'cpu',
                                 '--audits', '20', '--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)['audit']['failures'] == 0
    result = runner.invoke(app, ['bundle', 'build', str(honest), '--model', '1b-arch'])
    assert result.exit_code == 0 and (honest / 'manifest.json').exists()
    assert runner.invoke(app, ['validate', str(honest)]).exit_code == 0


def test_cli_init_copies_baseline(tmp_path):
    parent = tmp_path / 'parent'
    parent.mkdir()
    for name in ('config.json', 'tokenizer.json'):
        (parent / name).write_text('{}')
    save_file({'w': torch.ones(1)}, str(parent / 'model.safetensors'))
    out = tmp_path / 'mine'
    assert CliRunner().invoke(app, ['bundle', 'init', str(out), '--parent', str(parent)]).exit_code == 0
    assert {p.name for p in out.iterdir()} == {'delta_model.py', 'config.json', 'tokenizer.json', 'model.safetensors'}
    assert 'transformers' in (out / 'delta_model.py').read_text()
    assert CliRunner().invoke(app, ['bundle', 'init', str(tmp_path / 'v'), '--parent', str(parent),
                                    '--template', 'vllm']).exit_code == 0
    assert 'vllm' in (tmp_path / 'v' / 'delta_model.py').read_text()


def test_vm_worker_runs_harness_and_reports(honest, tmp_path):
    import hashlib
    from delta_cli.arch import worker
    from delta_cli.arch.sandbox import local_launcher
    files = []
    for path in sorted(honest.iterdir()):
        data = path.read_bytes()
        files.append({'name': path.name, 'url': path.as_uri(), 'sha256': hashlib.sha256(data).hexdigest(),
                      'size_bytes': len(data)})
    test = tmp_path / 'test.jsonl'
    test.write_text(json.dumps({'text': 'private held-out text\u2028with a separator'}, ensure_ascii=False) + '\n')
    spec = {'bundle': files, 'test': {'url': test.as_uri(), 'sha256': hashlib.sha256(test.read_bytes()).hexdigest()},
            'limits': {'audit_count': 10}, 'calibration_repeats': 2, 'min_matmul_tflops': 1}
    scrubs = []
    out = worker.run(spec, tmp_path / 'work', launcher=local_launcher, probe=lambda: 5.0, smoke=None,
                     scrub=lambda: scrubs.append(1))
    assert out['status'] == 'complete', out
    assert out['result']['splits']['test']['documents'] == 1
    assert len(scrubs) == 2  # Before every audit runner (two calibration repeats).
    assert len(out['calibration']['score_seconds']) == 2
    slow = worker.run(spec, tmp_path / 'work2', launcher=local_launcher, probe=lambda: 0.5, smoke=None)
    assert slow == {'status': 'infra_error', 'code': 'host_too_slow', 'matmul_tflops': 0.5}
    files[0]['sha256'] = '0' * 64
    assert worker.run(spec, tmp_path / 'work3', launcher=local_launcher, probe=None, smoke=None)['status'] == 'infra_error'
    (honest / 'delta_model.py').write_text(HONEST.replace('return list(text.encode())', 'return [999]'))
    data = (honest / 'delta_model.py').read_bytes()
    files[0].update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    bad = worker.run(spec, tmp_path / 'work4', launcher=local_launcher, probe=None, smoke=None)
    assert bad['status'] == 'rejected' and bad['code'] == 'invalid_tokens'


def test_sandbox_smoke_command_keeps_image_and_lockdown(tmp_path):
    from delta_cli.arch.worker import smoke_command, SMOKE
    from delta_cli.arch.sandbox import docker_command
    image = 'ghcr.io/x/delta-arch@sha256:' + 'c' * 64
    spec = {'sandbox': {'image': image, 'runtime': 'runsc'}}
    cmd = smoke_command(spec, tmp_path)
    real = docker_command(image, tmp_path, 'delta-arch-smoke', runtime='runsc')
    # Same flags and image as the scoring runner; only the program differs.
    assert cmd[:real.index(image) + 1] == real[:real.index(image) + 1]
    assert cmd[-3:] == [image, '-c', SMOKE] and cmd[cmd.index('--entrypoint') + 1] == 'python3'
    assert '--network' in cmd and '--runtime' in cmd


def test_worker_splits_test_documents_by_domain(honest, tmp_path):
    import hashlib
    from delta_cli.arch import worker
    from delta_cli.arch.bundle import pooled
    from delta_cli.arch.sandbox import local_launcher
    files = [{'name': p.name, 'url': p.as_uri(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
              'size_bytes': p.stat().st_size} for p in sorted(honest.iterdir())]
    test = tmp_path / 'test.jsonl'
    docs = [{'text': 'def f(x):\n    return x\n', 'domain': 'code', 'gpt2_targets': 9},
            {'text': 'Theorem. Every prime p > 2 is odd.', 'domain': 'math', 'gpt2_targets': 12},
            {'text': 'Another proof follows.', 'domain': 'math', 'gpt2_targets': 5}]
    test.write_text(''.join(json.dumps(d) + '\n' for d in docs))
    spec = {'bundle': files, 'test': {'url': test.as_uri(), 'sha256': hashlib.sha256(test.read_bytes()).hexdigest()},
            'limits': {'audit_count': 10}}
    out = worker.run(spec, tmp_path / 'work', launcher=local_launcher, probe=None, smoke=None, scrub=lambda: None)
    assert out['status'] == 'complete', out
    splits = out['result']['splits']
    assert sorted(splits) == ['test/code', 'test/math'] and splits['test/math']['documents'] == 2
    total = pooled(splits, 'test')
    assert total['documents'] == 3 and total['gpt2_targets'] == 26
    assert total['loss_per_gpt2_token'] == pytest.approx(
        (splits['test/code']['nll_sum_nats'] + splits['test/math']['nll_sum_nats']) / 26)


def test_bundle_check_reports_each_domain(honest, tmp_path):
    docs = tmp_path / 'dev.jsonl'
    docs.write_text(''.join(json.dumps({'text': t, 'domain': d, 'gpt2_targets': 4}) + '\n'
                            for t, d in [('alpha beta', 'web'), ('x = 1', 'code'), ('gamma', 'web')]))
    result = CliRunner().invoke(app, ['bundle', 'check', str(honest), '--docs', str(docs), '--device', 'cpu',
                                      '--audits', '5'])
    assert result.exit_code == 0, result.output
    assert 'loss/GPT-2 token' in result.output.splitlines()[0]
    assert any(l.strip().startswith('code ') for l in result.output.splitlines())
    assert any(l.strip().startswith('web ') for l in result.output.splitlines())
