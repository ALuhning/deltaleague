"""Sandboxed runner: loads the participant bundle and answers harness requests.

Runs inside the evaluation container (or a plain subprocess for local checks).
Nothing here is trusted by the harness: every response is a claim that is
validated outside this process. Model output written to stdout is redirected to
stderr so it cannot corrupt framing.
"""
import argparse
import ctypes
import importlib
import os
import sys
import time

from .protocol import PROTOCOL, ENTRY_MODULE, ENTRY_CLASS, read_frame, write_frame, pack_ints, unpack_ints, split


def _floats(tensor):
    import torch
    t = tensor.detach().to('cpu', torch.float32).contiguous()
    return ctypes.string_at(t.data_ptr(), t.numel() * 4)


def _contract(model):
    hello = {}
    for name in ('vocab_size', 'bos_id', 'eos_id', 'max_context'):
        value = getattr(model, name, None)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f'Model.{name} must be an int')
        hello[name] = value
    for name in ('encode', 'decode', 'token_logprobs', 'next_logits'):
        if not callable(getattr(model, name, None)):
            raise TypeError(f'Model.{name} must be callable')
    batch = getattr(model, 'batch_tokens', None)
    hello['batch_tokens'] = batch if isinstance(batch, int) and batch > 0 else max(hello['max_context'], 8192)
    return hello


def _score(model, hello, device, lengths, ids):
    import torch
    half = len(ids) // 2
    inputs, targets = split(ids[:half], lengths), split(ids[half:], lengths)
    windows = getattr(model, 'score_windows', None)
    if callable(windows):
        # Engines that batch internally take unpadded windows in one call.
        import array
        rows = windows(inputs, targets)
        if len(rows) != len(lengths) or any(len(r) != n for r, n in zip(rows, lengths)):
            raise ValueError('score_windows shape')
        return b''.join(array.array('f', [float(v) for v in r]).tobytes() for r in rows)
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    claims, batch = [None] * len(lengths), []

    def flush():
        width = lengths[batch[0]]
        pad = hello['bos_id']
        x = torch.tensor([inputs[i] + [pad] * (width - lengths[i]) for i in batch], device=device)
        y = torch.tensor([targets[i] + [pad] * (width - lengths[i]) for i in batch], device=device)
        out = model.token_logprobs(x, y)
        if tuple(out.shape) != (len(batch), width):
            raise ValueError('token_logprobs shape')
        out = out.detach().float().cpu()
        for row, i in enumerate(batch):
            claims[i] = out[row, :lengths[i]]
        batch.clear()

    for i in order:
        if batch and lengths[batch[0]] * (len(batch) + 1) > hello['batch_tokens']:
            flush()
        batch.append(i)
    if batch:
        flush()
    return b''.join(_floats(c) for c in claims)


def _next(model, hello, device, lengths, ids):
    import torch
    rows = []
    for prefix in split(ids, lengths):
        out = model.next_logits(torch.tensor([prefix], device=device))
        if tuple(out.shape) != (1, hello['vocab_size']):
            raise ValueError('next_logits shape')
        rows.append(_floats(out))
    return b''.join(rows)


def _flops(model, device, ids):
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        model.token_logprobs(x, y)
    return int(counter.get_total_flops())


def serve(bundle, device, inp, out):
    import torch
    sys.path.insert(0, os.path.abspath(bundle))
    start = time.monotonic()
    with torch.inference_mode():
        model = getattr(importlib.import_module(ENTRY_MODULE), ENTRY_CLASS)(bundle, device)
        hello = _contract(model)
        write_frame(out, {'op': 'ready', 'protocol': PROTOCOL, 'load_seconds': time.monotonic() - start, **hello})
        while True:
            header, payload = read_frame(inp)
            op = header.get('op')
            if op == 'exit':
                return
            if op == 'encode':
                texts = header['texts']
                encoded = [list(model.encode(t)) for t in texts]
                flat = [int(i) for ids in encoded for i in ids]
                write_frame(out, {'op': 'encoded', 'lengths': [len(e) for e in encoded]}, pack_ints(flat))
            elif op == 'decode':
                texts = [model.decode(ids) for ids in split(unpack_ints(payload), header['lengths'])]
                if not all(isinstance(t, str) for t in texts):
                    raise TypeError('decode must return str')
                write_frame(out, {'op': 'decoded', 'texts': texts})
            elif op == 'score':
                write_frame(out, {'op': 'scored'}, _score(model, hello, device, header['lengths'], unpack_ints(payload)))
            elif op == 'next':
                write_frame(out, {'op': 'rows'}, _next(model, hello, device, header['lengths'], unpack_ints(payload)))
            elif op == 'flops':
                write_frame(out, {'op': 'flops', 'flops': _flops(model, device, unpack_ints(payload))})
            else:
                raise ValueError('unknown op')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    # Protocol streams are private duplicates; fd 0/1 become inert for model code.
    out = os.fdopen(os.dup(1), 'wb')
    inp = os.fdopen(os.dup(0), 'rb')
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        serve(args.bundle, args.device, inp, out)
    except Exception as exc:  # The class name only: messages may echo scored text.
        import traceback
        traceback.print_exc()  # stderr stays with the runner's operator, never the API.
        try:
            write_frame(out, {'op': 'error', 'type': type(exc).__name__})
        except Exception:
            pass
        raise SystemExit(1)


if __name__ == '__main__':
    main()
