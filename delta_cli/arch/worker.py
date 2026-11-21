"""Official-scoring orchestrator on a rented GPU VM (trusted; outside the sandbox).

Started by the platform with a capability URL to ``input.json`` and its digest.
It downloads the bundle and documents, checks the host and the sandbox runtime,
runs the harness with a fresh container per phase, and uploads one
``output.json``. Sandbox stdout/stderr are discarded: they could contain scored
text. The VM holds no platform credentials, only this job's expiring URLs.
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from .harness import ArchEvalError, Limits, evaluate
from .sandbox import docker_command, docker_launcher

WORK = Path(os.environ.get('DELTA_ARCH_WORK', '/tmp/delta-arch'))


def fetch(url, sha256=None, limit=1024**3):
    # Chunked: read(n) preallocates n bytes on HTTP and file responses.
    chunks, size = [], 0
    with urllib.request.urlopen(url, timeout=600) as response:
        while chunk := response.read(8 << 20):
            size += len(chunk)
            if size > limit:
                raise ValueError('download too large')
            chunks.append(chunk)
    data = b''.join(chunks)
    if sha256 and hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError('download digest mismatch')
    return data


def fetch_file(url, target, sha256, size):
    h, n = hashlib.sha256(), 0
    with urllib.request.urlopen(url, timeout=600) as response, open(target, 'wb') as out:
        while chunk := response.read(8 << 20):
            n += len(chunk)
            if n > size:
                raise ValueError('download too large')
            h.update(chunk)
            out.write(chunk)
    if n != size or h.hexdigest() != sha256:
        raise ValueError('download digest mismatch')


def upload(url, data):
    request = urllib.request.Request(url, data=data, method='PUT')
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120):
                return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def matmul_tflops(seconds=3.0):
    """Trusted host-speed probe: dense bf16 matmul throughput."""
    import torch
    a = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device='cuda', dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    start, n = time.perf_counter(), 0
    while time.perf_counter() - start < seconds:
        a @ b
        n += 1
    torch.cuda.synchronize()
    return 2 * 8192**3 * n / (time.perf_counter() - start) / 1e12


def scrub_gpu(reserve=256 * 1024**2):
    """Overwrite free device memory, so the audit runner cannot read what phase A left behind."""
    import torch
    if not torch.cuda.is_available():
        return 0
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    size = max(0, free - reserve)
    block = torch.zeros(size, dtype=torch.uint8, device='cuda')
    torch.cuda.synchronize()
    del block
    torch.cuda.empty_cache()
    return size


def scrubbed(launch, scrub):
    def wrapped(phase):
        if phase == 'audit' and scrub:
            scrub()
        return launch(phase)
    return wrapped


def sandbox_smoke(spec, bundle):
    """The locked-down runtime must start and see the GPU before any participant code runs."""
    sandbox = spec['sandbox']
    command = docker_command(sandbox['image'], bundle, 'delta-arch-smoke', runtime=sandbox.get('runtime'))
    command[command.index('--entrypoint') + 2:] = ['-c', 'import torch; assert torch.cuda.is_available()']
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    if result.returncode:
        raise RuntimeError('sandbox_unavailable')


def prepare(spec, work):
    bundle = work / 'bundle'
    bundle.mkdir(parents=True)
    for f in spec['bundle']:
        fetch_file(f['url'], bundle / f['name'], f['sha256'], f['size_bytes'])
        (bundle / f['name']).chmod(0o444)
    bundle.chmod(0o555)
    splits = {}
    validation = spec.get('validation')
    if validation:
        from .bundle import fineweb_docs
        splits['validation'] = fineweb_docs(validation['documents'])
    test = spec.get('test')
    if test:
        raw = fetch(test['url'], test['sha256'])
        splits['test'] = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return bundle, splits


def run(spec, work, launcher=None, probe=matmul_tflops, smoke=sandbox_smoke, scrub=scrub_gpu):
    """Returns the output document. Never raises for model-caused failures."""
    try:
        bundle, splits = prepare(spec, work)
        tflops = probe() if probe else None
        floor = spec.get('min_matmul_tflops')
        if floor and tflops is not None and tflops < floor:
            return {'status': 'infra_error', 'code': 'host_too_slow', 'matmul_tflops': tflops}
        if smoke:
            smoke(spec, bundle)
    except Exception as exc:
        return {'status': 'infra_error', 'code': str(exc) if str(exc) == 'sandbox_unavailable' else type(exc).__name__}
    sandbox = spec.get('sandbox', {})
    launch = launcher(bundle) if launcher else docker_launcher(sandbox['image'], bundle, runtime=sandbox.get('runtime'))
    launch = scrubbed(launch, scrub)
    limits = Limits(**spec.get('limits', {}))
    repeats = max(1, int(spec.get('calibration_repeats', 1)))
    timings, result = [], None
    try:
        for _ in range(repeats):
            result = evaluate(launch, splits, limits)
            timings.append(result['score_seconds'])
    except ArchEvalError as exc:
        return {'status': 'rejected', 'code': exc.code, 'detail': exc.detail[:200], 'stats': exc.stats,
                'matmul_tflops': tflops}
    except Exception as exc:
        return {'status': 'infra_error', 'code': type(exc).__name__}
    output = {'status': 'complete', 'result': result, 'matmul_tflops': tflops}
    if repeats > 1:
        output['calibration'] = {'score_seconds': timings, 'median_score_seconds': statistics.median(timings)}
    return output


def main(argv=None):
    input_url, input_sha = (argv or sys.argv[1:])[:2]
    spec = json.loads(fetch(input_url, input_sha))
    upload(spec['status_url'], json.dumps({'stage': 'running', 'at': time.time()}).encode())
    output = run(spec, WORK)
    upload(spec['output_url'], json.dumps(output, allow_nan=False).encode())


if __name__ == '__main__':
    main()
