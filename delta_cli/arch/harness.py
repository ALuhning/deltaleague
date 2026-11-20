"""Trusted scoring harness for architecture-track bundles.

Phase A (timed) runs the bundle on every document and collects one claimed
log-probability per target token. Phase B starts a *fresh* runner that has never
seen the text and, only after phase A has committed its claims:

1. audits randomly chosen target positions: the runner sees just the prefix and
   returns a full logit row; the harness normalizes it itself and compares the
   target's log-probability with phase A's claim;
2. measures FLOPs per token on a fixed probe;
3. checks that ``decode(encode(text)) == text`` for every document.

Each honest claim is then a normalized next-token probability that depends only
on the prefix, and the token sequence decodes uniquely to the text, so the summed
negative log-likelihood upper-bounds -log P(text) whatever the tokenizer does.
A systematic positive bias that stays within the per-token tolerance is charged
back as a penalty on every target token, so it cannot improve a score.
"""
from array import array
from bisect import bisect_right
from dataclasses import dataclass, asdict
import math
import random
import secrets
import threading
import time

from .protocol import PROTOCOL, ProtocolError, read_frame, write_frame, pack_ints, unpack_ints, split

CONTRACT = ('vocab_size', 'bos_id', 'eos_id', 'max_context')
ERRORS = {
    'load_timeout': 'The model did not become ready within the load limit.',
    'time_limit_exceeded': 'Scoring exceeded the time limit.',
    'audit_timeout': 'The audit runner exceeded its time limit.',
    'model_error': 'The model raised an exception.',
    'protocol_error': 'The runner broke the harness protocol.',
    'invalid_contract': 'Model attributes are outside the contract.',
    'invalid_tokens': 'encode returned token ids outside [0, vocab_size).',
    'invalid_claims': 'token_logprobs returned non-finite or positive log-probabilities.',
    'audit_failed': 'Claimed log-probabilities differ from prefix-only recomputation.',
    'roundtrip_failed': 'decode(encode(text)) did not reproduce the text.',
    'flops_exceeded': 'Measured FLOPs per token exceed the cap.',
}


class ArchEvalError(Exception):
    def __init__(self, code, detail='', stats=None):
        super().__init__(code)
        self.code, self.detail, self.stats = code, detail, stats or {}


@dataclass(frozen=True)
class Limits:
    time_limit_seconds: float | None = None  # phase A, from ready to last claim
    load_seconds: float = 900
    audit_seconds: float | None = None       # whole phase B after ready
    audit_count: int = 1000
    audit_tolerance: float = 0.25            # nats, per audited token
    flops_per_token_cap: float | None = None
    flops_probe_tokens: int = 4096
    max_vocab: int = 1 << 21
    max_context: int = 1 << 20
    request_tokens: int = 1 << 18
    encode_chars: int = 4 << 20
    audit_batch: int = 16


class Runner:
    """One sandboxed process. A watchdog kills it when the current deadline passes."""

    def __init__(self, process, seconds, code):
        self.process, self.timer, self.expired = process, None, None
        self.deadline(seconds, code)

    def deadline(self, seconds, code):
        if self.timer:
            self.timer.cancel()
        self.timer = None
        if seconds is not None:
            self.timer = threading.Timer(seconds, self._expire, (code,))
            self.timer.daemon = True
            self.timer.start()

    def _expire(self, code):
        self.expired = code
        self.kill()

    def kill(self):
        hook = getattr(self.process, 'delta_kill', None)
        if hook:
            try:
                hook()
            except Exception:
                pass
        try:
            self.process.kill()
        except OSError:
            pass

    def receive(self):
        try:
            header, payload = read_frame(self.process.stdout)
        except (ProtocolError, OSError, ValueError) as exc:
            if self.expired:
                raise ArchEvalError(self.expired) from None
            code = 'model_error' if self.process.poll() not in (None, 0) else 'protocol_error'
            raise ArchEvalError(code, type(exc).__name__) from None
        if header.get('op') == 'error':
            raise ArchEvalError('model_error', str(header.get('type', ''))[:100])
        return header, payload

    def request(self, header, payload=b''):
        try:
            write_frame(self.process.stdin, header, payload)
        except (OSError, ValueError):
            if self.expired:
                raise ArchEvalError(self.expired) from None
            raise ArchEvalError('model_error', 'runner exited') from None
        return self.receive()

    def ready(self, limits):
        header, _ = self.receive()
        if header.get('op') != 'ready' or header.get('protocol') != PROTOCOL:
            raise ArchEvalError('protocol_error', 'ready')
        hello = {k: header.get(k) for k in CONTRACT}
        hello['batch_tokens'] = header.get('batch_tokens')
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in hello.values()):
            raise ArchEvalError('invalid_contract')
        if not (2 <= hello['vocab_size'] <= limits.max_vocab and 2 <= hello['max_context'] <= limits.max_context
                and 0 <= hello['bos_id'] < hello['vocab_size'] and 0 <= hello['eos_id'] < hello['vocab_size']):
            raise ArchEvalError('invalid_contract')
        return hello

    def close(self):
        if self.timer:
            self.timer.cancel()
        try:
            write_frame(self.process.stdin, {'op': 'exit'})
            self.process.stdin.close()
            self.process.wait(timeout=30)
        except Exception:
            self.kill()
        finally:
            self.process.wait()


def _batches(items, size_of, budget):
    batch, used = [], 0
    for item in items:
        n = size_of(item)
        if batch and used + n > budget:
            yield batch
            batch, used = [], 0
        batch.append(item)
        used += n
    if batch:
        yield batch


class Split:
    """Phase A record for one document set: window targets and claims."""

    def __init__(self, name, docs):
        self.name, self.docs = name, docs
        self.ids = []            # token ids per document (no eos)
        self.windows = []        # (doc index, array of targets)
        self.claims = []         # array('d') per window
        self.starts = []         # global target offset of each window

    @property
    def targets(self):
        return sum(len(t) for _, t in self.windows)


def _encode(runner, split_, hello, limits):
    vocab = hello['vocab_size']
    for batch in _batches(range(len(split_.docs)), lambda i: len(split_.docs[i]['text']) + 1, limits.encode_chars):
        header, payload = runner.request({'op': 'encode', 'texts': [split_.docs[i]['text'] for i in batch]})
        lengths = header.get('lengths')
        if header.get('op') != 'encoded' or not isinstance(lengths, list) or len(lengths) != len(batch):
            raise ArchEvalError('protocol_error', 'encode')
        try:
            pieces = split(unpack_ints(payload), lengths)
        except ProtocolError:
            raise ArchEvalError('protocol_error', 'encode') from None
        for ids in pieces:
            if any(not 0 <= t < vocab for t in ids):
                raise ArchEvalError('invalid_tokens')
            split_.ids.append(array('i', ids))


def _windows(split_, hello):
    step = hello['max_context']
    for d, (doc, ids) in enumerate(zip(split_.docs, split_.ids)):
        targets = array('i', ids)
        if doc.get('predicts_eos', True):
            targets.append(hello['eos_id'])
        for i in range(0, len(targets), step):
            split_.windows.append((d, targets[i:i + step]))


def _inputs(hello, targets):
    return [hello['bos_id'], *targets[:-1]]


def _score(runner, split_, hello, limits):
    split_.claims = [None] * len(split_.windows)
    for batch in _batches(range(len(split_.windows)), lambda i: len(split_.windows[i][1]), limits.request_tokens):
        targets = [split_.windows[i][1] for i in batch]
        flat = [t for w in targets for t in _inputs(hello, w)] + [t for w in targets for t in w]
        header, payload = runner.request({'op': 'score', 'lengths': [len(w) for w in targets]}, pack_ints(flat))
        if header.get('op') != 'scored' or len(payload) != 4 * (len(flat) // 2):
            raise ArchEvalError('protocol_error', 'score')
        values = array('f')
        values.frombytes(payload)
        at = 0
        for i, w in zip(batch, targets):
            claims = array('d', values[at:at + len(w)])
            at += len(w)
            if not all(math.isfinite(c) and c <= 1e-4 for c in claims):
                raise ArchEvalError('invalid_claims')
            split_.claims[i] = claims


def phase_a(launch, splits, limits):
    runner = Runner(launch('score'), limits.load_seconds, 'load_timeout')
    try:
        hello = runner.ready(limits)
        load = time.monotonic()
        runner.deadline(limits.time_limit_seconds, 'time_limit_exceeded')
        records = []
        for name, docs in splits.items():
            record = Split(name, docs)
            _encode(runner, record, hello, limits)
            _windows(record, hello)
            _score(runner, record, hello, limits)
            records.append(record)
        elapsed = time.monotonic() - load
        if limits.time_limit_seconds is not None and elapsed > limits.time_limit_seconds:
            raise ArchEvalError('time_limit_exceeded')
        return hello, records, elapsed
    finally:
        runner.close()


def _audit_plan(records, count, rng):
    index, total = [], 0
    for r, record in enumerate(records):
        for w, (_, targets) in enumerate(record.windows):
            index.append((total, r, w))
            total += len(targets)
    starts = [i[0] for i in index]
    picks = []
    for g in rng.sample(range(total), min(count, total)) if total else []:
        _, r, w = index[bisect_right(starts, g) - 1]
        picks.append((r, records[r].windows[w][0], w, g - starts[bisect_right(starts, g) - 1]))
    # Per document, a prefix is only ever followed by longer prefixes, so the
    # runner has answered every audited position before it sees that target.
    groups = {}
    for pick in sorted(picks):
        groups.setdefault(pick[:2], []).append(pick)
    rounds, k = [], 0
    while True:
        layer = [g[k] for g in groups.values() if len(g) > k]
        if not layer:
            return rounds
        rounds.append(layer)
        k += 1


def _audit(runner, records, hello, limits, rng):
    deviations = []
    for layer in _audit_plan(records, limits.audit_count, rng):
        for batch in _batches(layer, lambda _: 1, limits.audit_batch):
            prefixes = []
            for r, _, w, p in batch:
                targets = records[r].windows[w][1]
                prefixes.append(_inputs(hello, targets)[:p + 1])
            header, payload = runner.request({'op': 'next', 'lengths': [len(x) for x in prefixes]},
                                             pack_ints([t for x in prefixes for t in x]))
            if header.get('op') != 'rows' or len(payload) != 4 * hello['vocab_size'] * len(batch):
                raise ArchEvalError('protocol_error', 'next')
            import torch
            rows = torch.frombuffer(bytearray(payload), dtype=torch.float32).view(len(batch), hello['vocab_size'])
            logprobs = torch.log_softmax(rows.double(), dim=-1)
            for row, (r, _, w, p) in enumerate(batch):
                target = records[r].windows[w][1][p]
                audited = float(logprobs[row, target])
                claim = records[r].claims[w][p]
                deviations.append(claim - audited if math.isfinite(audited) else math.inf)
    return deviations


def _decode(runner, record, limits):
    for batch in _batches(range(len(record.docs)), lambda i: len(record.ids[i]) + 1, limits.request_tokens):
        header, _ = runner.request({'op': 'decode', 'lengths': [len(record.ids[i]) for i in batch]},
                                   pack_ints([t for i in batch for t in record.ids[i]]))
        texts = header.get('texts')
        if header.get('op') != 'decoded' or not isinstance(texts, list) or len(texts) != len(batch):
            raise ArchEvalError('protocol_error', 'decode')
        if any(text != record.docs[i]['text'] for i, text in zip(batch, texts)):
            raise ArchEvalError('roundtrip_failed')


def _probe(hello, n, seed):
    rng = random.Random(seed)
    n = max(2, min(n, hello['max_context']) + 1)
    return [hello['bos_id']] + [rng.randrange(hello['vocab_size']) for _ in range(n - 1)]


def phase_b(launch, hello, records, limits, seed):
    runner = Runner(launch('audit'), limits.load_seconds, 'load_timeout')
    try:
        again = runner.ready(limits)
        if any(again[k] != hello[k] for k in CONTRACT):
            raise ArchEvalError('invalid_contract', 'contract changed between runners')
        start = time.monotonic()
        runner.deadline(limits.audit_seconds, 'audit_timeout')
        deviations = _audit(runner, records, hello, limits, random.Random(seed))
        probe = _probe(hello, limits.flops_probe_tokens, seed)
        header, _ = runner.request({'op': 'flops'}, pack_ints(probe))
        flops = header.get('flops')
        if header.get('op') != 'flops' or not isinstance(flops, int) or flops < 0:
            raise ArchEvalError('protocol_error', 'flops')
        for record in records:
            _decode(runner, record, limits)
        return deviations, flops / (len(probe) - 1), time.monotonic() - start
    finally:
        runner.close()


def summarize(deviations, tolerance):
    k = len(deviations)
    mean = sum(deviations) / k if k else 0.0
    std = math.sqrt(sum((d - mean) ** 2 for d in deviations) / (k - 1)) if k > 1 else 0.0
    worst = max((abs(d) for d in deviations), default=0.0)
    return {'count': k, 'mean_deviation': mean, 'std_deviation': std, 'max_abs_deviation': worst,
            'tolerance': tolerance, 'failures': sum(1 for d in deviations if not abs(d) <= tolerance),
            'penalty_nats_per_token': max(0.0, mean)}


def metrics(record, penalty):
    claims = math.fsum(math.fsum(c) for c in record.claims)
    targets = record.targets
    nll = -claims + penalty * targets
    utf8 = sum(len(d['text'].encode()) for d in record.docs)
    out = {'documents': len(record.docs), 'targets': targets, 'windows': len(record.windows),
           'utf8_bytes': utf8, 'nll_sum_nats': nll,
           'nats_per_token': nll / targets if targets else None,
           'bits_per_byte': nll / math.log(2) / utf8 if utf8 else None}
    if record.docs and all('gpt2_targets' in d for d in record.docs):
        gpt2 = sum(d['gpt2_targets'] for d in record.docs)
        out.update(gpt2_targets=gpt2, loss_per_gpt2_token=nll / gpt2 if gpt2 else None)
    return out


def evaluate(launch, splits, limits=Limits(), seed=None):
    """Score ``splits`` ({name: [{'text', 'predicts_eos'?, 'gpt2_targets'?}]}).

    ``launch(phase)`` returns a fresh runner process (``subprocess.Popen`` with
    binary stdin/stdout pipes) for phase ``'score'`` or ``'audit'``.
    Raises ArchEvalError with a stable ``code`` on any rejection.
    """
    hello, records, score_seconds = phase_a(launch, splits, limits)
    stats = {'score_seconds': score_seconds, 'contract': {k: hello[k] for k in CONTRACT}}
    # Drawn only after every claim is committed.
    seed = secrets.randbits(63) if seed is None else seed
    try:
        deviations, flops_per_token, audit_seconds = phase_b(launch, hello, records, limits, seed)
    except ArchEvalError as exc:
        exc.stats = {**stats, **exc.stats}
        raise
    audit = summarize(deviations, limits.audit_tolerance)
    stats.update(audit={**audit, 'seed': seed}, audit_seconds=audit_seconds, flops_per_token=flops_per_token)
    if audit['failures']:
        raise ArchEvalError('audit_failed', f"{audit['failures']} of {audit['count']} positions", stats)
    if limits.flops_per_token_cap is not None and flops_per_token > limits.flops_per_token_cap:
        raise ArchEvalError('flops_exceeded', f'{flops_per_token:.4g} > {limits.flops_per_token_cap:.4g}', stats)
    return {**stats, 'limits': asdict(limits),
            'splits': {r.name: metrics(r, audit['penalty_nats_per_token']) for r in records}}
