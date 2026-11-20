"""Length-prefixed frames between the trusted harness and the sandboxed runner.

A frame is a 4-byte big-endian header length, a UTF-8 JSON header and, when the
header names ``nbytes``, that many raw payload bytes. Integer payloads are
little-endian int32; float payloads are little-endian float32.
"""
import json
import struct

PROTOCOL = 'delta-arch-v1'
ENTRY_MODULE = 'delta_model'
ENTRY_CLASS = 'Model'
MAX_HEADER = 64 * 1024**2
MAX_PAYLOAD = 8 * 1024**3


class ProtocolError(Exception):
    """The peer broke framing or sent an out-of-contract response."""


def write_frame(stream, header, payload=b''):
    header = dict(header, nbytes=len(payload))
    raw = json.dumps(header, separators=(',', ':'), allow_nan=False).encode()
    stream.write(struct.pack('>I', len(raw)) + raw)
    if payload:
        stream.write(payload)
    stream.flush()


def _exact(stream, n):
    chunks, left = [], n
    while left:
        chunk = stream.read(left)
        if not chunk:
            raise ProtocolError('stream closed')
        chunks.append(chunk)
        left -= len(chunk)
    return b''.join(chunks)


def read_frame(stream):
    (size,) = struct.unpack('>I', _exact(stream, 4))
    if size > MAX_HEADER:
        raise ProtocolError('header too large')
    try:
        header = json.loads(_exact(stream, size))
    except ValueError as exc:
        raise ProtocolError('invalid header') from exc
    if not isinstance(header, dict):
        raise ProtocolError('invalid header')
    nbytes = header.get('nbytes', 0)
    if not isinstance(nbytes, int) or not 0 <= nbytes <= MAX_PAYLOAD:
        raise ProtocolError('invalid payload size')
    return header, _exact(stream, nbytes) if nbytes else b''


def pack_ints(values):
    import array
    data = array.array('i', values)
    if data.itemsize != 4:
        raise RuntimeError('int32 array support required')
    return data.tobytes() if struct.pack('=i', 1) == struct.pack('<i', 1) else _swap(data)


def unpack_ints(payload):
    import array
    if len(payload) % 4:
        raise ProtocolError('misaligned int32 payload')
    data = array.array('i')
    data.frombytes(payload)
    if struct.pack('=i', 1) != struct.pack('<i', 1):
        data.byteswap()
    return data.tolist()


def _swap(data):
    data = data.__copy__()
    data.byteswap()
    return data.tobytes()


def split(values, lengths):
    out, at = [], 0
    for n in lengths:
        out.append(values[at:at + n])
        at += n
    if at != len(values):
        raise ProtocolError('payload length mismatch')
    return out
