"""Bundle inventory, manifest and safetensors checks. Never imports participant code."""
import json
import shutil
from pathlib import Path
from importlib.resources import files as package_files

from ..artifacts import (BUNDLE_SCHEMA, BUNDLE_SUFFIXES, BUNDLE_RESERVED, file_hash, validate_bundle_manifest,
                         bundle_hash)

BASELINE_ASSETS = ('config.json', 'tokenizer.json')


def scan(root):
    root = Path(root)
    found = []
    for path in sorted(root.iterdir()):
        if path.name in BUNDLE_RESERVED or path.name.startswith('.') or path.name == '__pycache__':
            continue
        if path.is_dir() or path.is_symlink() or not path.name.endswith(BUNDLE_SUFFIXES):
            raise ValueError(f'{path.name}: bundles hold flat regular files with suffixes {", ".join(BUNDLE_SUFFIXES)}')
        found.append({'name': path.name, 'sha256': file_hash(path), 'size_bytes': path.stat().st_size})
    return found


def build_manifest(root, model_config_id, parent_version_id=None):
    manifest = {'schema_version': BUNDLE_SCHEMA, 'format': 'bundle', 'model_config_id': model_config_id,
                'artifacts': scan(root)}
    if parent_version_id:
        manifest['parent_version_id'] = parent_version_id
    validate_bundle_manifest(manifest)
    check_safetensors(root, manifest['artifacts'])
    (Path(root) / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def check_safetensors(root, artifacts, finite=True):
    """Parse headers (no pickle) and, optionally, check every float tensor is finite."""
    from safetensors import safe_open
    import torch
    for artifact in artifacts:
        if not artifact['name'].endswith('.safetensors'):
            continue
        with safe_open(str(Path(root) / artifact['name']), framework='pt') as handle:
            for key in handle.keys():
                if not finite:
                    continue
                tensor = handle.get_tensor(key)
                if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f'non-finite tensor: {key}')


def init(output, parent=None):
    """Scaffold a bundle from the baseline code, copying a pulled parent's config, tokenizer and weights."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy(package_files('delta_cli.arch.baseline') / 'delta_model.py', output / 'delta_model.py')
    if parent:
        parent = Path(parent)
        for name in BASELINE_ASSETS:
            shutil.copy(parent / name, output / name)
        for weights in sorted(parent.glob('*.safetensors')):
            shutil.copy(weights, output / weights.name)
    return output


def identity(manifest):
    return bundle_hash(manifest['artifacts'])


def load_docs(path, limit=None):
    """JSONL documents: {"text", "predicts_eos"?, "gpt2_targets"?}."""
    docs = []
    with open(path) as f:
        for line in f:
            if limit is not None and len(docs) >= limit:
                break
            doc = json.loads(line)
            if not isinstance(doc.get('text'), str):
                raise ValueError('each document needs a text string')
            docs.append(doc)
    return docs


def fineweb_docs(limit=None):
    """Public validation: the pinned modded-nanogpt FineWeb window, as harness documents."""
    from ..public_eval import fineweb_loss
    docs = fineweb_loss.load_documents((0, limit) if limit else None)
    return [{'text': d['text'], 'predicts_eos': d['predicts_eot'],
             'gpt2_targets': d['gpt2_tokens'] + int(d['predicts_eot'])} for d in docs]

