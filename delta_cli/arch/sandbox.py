"""Runner launchers.

``local_launcher`` is a plain subprocess for ``delta bundle check``: it is not a
security boundary and a local runner can keep state between phases. Official
scoring uses ``docker_launcher``: every phase is a fresh container with no
network, a read-only root and bundle, no capabilities and an empty tmpfs, so the
audit runner cannot see anything phase A saw.
"""
import os
import subprocess
import sys
import tempfile
import uuid

INNER = ['-I', '-B', '-m', 'delta_cli.arch.inner']


def _stderr(target):
    # 'inherit' shows runner output to a local user; the platform discards it.
    return None if target == 'inherit' else (target if target is not None else subprocess.DEVNULL)


def local_launcher(bundle, device='cpu', python=None, stderr=None, env=None):
    bundle = os.path.abspath(bundle)

    def launch(phase):
        scratch = tempfile.mkdtemp(prefix=f'delta-arch-{phase}-')
        child = {'PATH': os.environ.get('PATH', ''), 'HOME': scratch, 'TMPDIR': scratch,
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONHASHSEED': '0'}
        for name in ('CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH'):
            if name in os.environ:
                child[name] = os.environ[name]
        child.update(env or {})
        return subprocess.Popen([python or sys.executable, *INNER, '--bundle', bundle, '--device', device],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=_stderr(stderr), env=child, cwd=scratch)
    return launch


def docker_command(image, bundle, name, *, runtime=None, gpu='0', memory='64g', shm='16g', pids=1024,
                   tmpfs='16g', cpus=None):
    if '@sha256:' not in image:
        raise ValueError('Sandbox image must be digest-pinned')
    command = ['docker', 'run', '--rm', '-i', '--name', name, '--network', 'none', '--read-only',
               '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--user', '65534:65534',
               '--pids-limit', str(pids), '--memory', memory, '--memory-swap', memory, '--shm-size', shm,
               '--ipc', 'private', '--tmpfs', f'/tmp:rw,nosuid,nodev,size={tmpfs}',
               '-e', 'HOME=/tmp', '-e', 'TMPDIR=/tmp', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
               '-e', 'PYTHONHASHSEED=0',
               '--mount', f'type=bind,src={os.path.abspath(bundle)},dst=/bundle,readonly',
               '--gpus', f'device={gpu}']
    if runtime:
        command += ['--runtime', runtime]
    if cpus:
        command += ['--cpus', str(cpus)]
    return command + ['--entrypoint', 'python3', image, *INNER, '--bundle', '/bundle', '--device', 'cuda']


def docker_launcher(image, bundle, stderr=None, **options):
    def launch(phase):
        name = f'delta-arch-{phase}-{uuid.uuid4().hex[:12]}'
        process = subprocess.Popen(docker_command(image, bundle, name, **options), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=_stderr(stderr))
        process.delta_kill = lambda: subprocess.run(['docker', 'kill', name], stdout=subprocess.DEVNULL,
                                                    stderr=subprocess.DEVNULL, timeout=30)
        return process
    return launch
