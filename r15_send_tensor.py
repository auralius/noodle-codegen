#!/usr/bin/env python3
"""Noodle Studio R15 Serial IMG v1 sender for generated AVR firmware.

Transport v1 is byte-oriented: the board expects raw uint8 bytes; the generated
firmware is responsible for its configured transform to the model's tensor type.
No shape/dtype is sent over the wire: those are build-time firmware contracts.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

PROTOCOL = 'noodle-serial-img-v1'


def parse_shape(text: str) -> list[int]:
    try:
        shape = [int(v.strip()) for v in text.replace('x', ',').split(',')]
    except ValueError as ex:
        raise ValueError(f'invalid shape {text!r}: expected e.g. 1,256 or 1,16,16,1') from ex
    if not shape or any(x <= 0 for x in shape):
        raise ValueError('shape dimensions must be positive integers')
    return shape


def numel(shape: list[int]) -> int:
    n = 1
    for x in shape:
        n *= x
    return n


def payload_for(n: int, pattern: str, value: int, seed: int, infile: str | None) -> bytes:
    if infile is not None:
        data = Path(infile).read_bytes()
        if len(data) != n:
            raise ValueError(f'input file size {len(data)} bytes, but firmware expects {n} bytes')
        return data
    if pattern == 'zeros':
        return bytes(n)
    if pattern == 'ramp':
        return bytes(i & 255 for i in range(n))
    if pattern == 'constant':
        if not 0 <= value <= 255:
            raise ValueError('--value must be in 0..255')
        return bytes([value]) * n
    if pattern == 'random':
        rng = random.Random(seed)
        return bytes(rng.randrange(256) for _ in range(n))
    raise ValueError('unknown pattern')


def read_line(ser, deadline: float, max_len: int = 1024) -> str | None:
    """Read a line safely (some AVR reset sequences produce binary startup noise)."""
    buf = bytearray()
    while time.monotonic() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b == b'\n':
            result = buf.decode('ascii', errors='replace').strip()
            return result
        if b == b'\r':
            continue
        buf.extend(b)
        if len(buf) > max_len:
            buf.clear()   # ignore corrupted startup bytes instead of printing them
    return None


def await_token_line(ser, wanted: str, timeout_s: float, show: bool = True) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        line = read_line(ser, deadline)
        if line is None:
            break
        if not line:
            continue
        if show and len(line) < 160:
            print(f'[rx] {line}', flush=True)
        if line == wanted:
            return
        if line.startswith('ERR_'):
            raise RuntimeError(f'board reported {line}')
    raise TimeoutError(f'timed out waiting for {wanted!r}; check firmware, port and reset')


def await_ack(ser, timeout_s: float) -> None:
    """Each 32-byte block receives a single 'K' byte WITHOUT a newline."""
    deadline = time.monotonic() + timeout_s
    noise = bytearray()
    while time.monotonic() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b == b'K':
            return
        noise.extend(b)
        if len(noise) > 256:
            noise = noise[-128:]
        if b'ERR_INPUT' in noise or b'ERR_INFERENCE' in noise:
            raise RuntimeError('MCU reported input/inference error during transfer')
    raise TimeoutError(f'no K acknowledgement after chunk; recent bytes={bytes(noise)!r}')


def send_one(ser, payload: bytes, chunk_size: int, timeout_s: float, inference_timeout: float):
    ser.write(b'IMG\n')
    ser.flush()
    await_token_line(ser, 'RDYIMG', timeout_s)
    t_tx = time.monotonic()
    for i in range(0, len(payload), chunk_size):
        block = payload[i:i + chunk_size]
        ser.write(block)
        ser.flush()
        # Current firmware sends K *only* after full 32-byte groups.
        if len(block) == chunk_size:
            await_ack(ser, timeout_s)
    print(f'[tx] {len(payload)} bytes, {len(payload) // chunk_size} acknowledged blocks'
          f'{" plus tail" if len(payload) % chunk_size else ""}, transfer={time.monotonic()-t_tx:.3f}s')
    predicted = None
    output = None
    board_ms = None
    deadline = time.monotonic() + inference_timeout
    while time.monotonic() < deadline:
        line = read_line(ser, deadline)
        if line is None:
            break
        if not line:
            continue
        print(f'[rx] {line}', flush=True)
        if line.startswith('PRED '):
            m = re.fullmatch(r'PRED\s+(\d+)\s+MS\s+(\d+)', line)
            if m:
                predicted = int(m.group(1))
                board_ms = int(m.group(2))
        elif line.startswith('OUTPUT_INT8 '):
            output = [int(x) for x in line.split()[1:]]
        elif line == 'READY':
            if predicted is None:
                raise RuntimeError('board returned READY without a valid PRED line')
            return predicted, board_ms, output
        elif line.startswith('ERR_'):
            raise RuntimeError(f'board reported {line}')
    raise TimeoutError('timed out waiting for MCU inference result')


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='Send a dummy/input byte tensor to R15 Arduino Uno using IMG/K serial v1.')
    p.add_argument('--port', help='Serial port, e.g. /dev/ttyUSB0 or /dev/ttyACM0')
    p.add_argument('--manifest', help='Optional input_spec.json exported by Codegen')
    p.add_argument('--shape', help='Tensor shape (e.g. 1,256 or 1,16,16,1); no shape is sent over serial')
    p.add_argument('--elements', type=int, help='Expected raw byte count; overrides shape/manifest')
    p.add_argument('--baud', type=int, help='Baud rate (default 115200 or manifest)')
    p.add_argument('--chunk', type=int, help='ACK chunk length (default 32 or manifest; must match firmware)')
    p.add_argument('--pattern', choices=['zeros', 'ramp', 'random', 'constant'], default='ramp')
    p.add_argument('--value', type=int, default=128, help='Pixel byte for --pattern constant (0..255)')
    p.add_argument('--seed', type=int, default=1234, help='Seed for random pattern, making repeat runs identical')
    p.add_argument('--input-file', help='Existing raw uint8 binary data, exactly N bytes')
    p.add_argument('--repeat', type=int, default=1, help='Number of times to resend identical payload')
    p.add_argument('--timeout', type=float, default=5.0, help='Handshake/ACK timeout, seconds')
    p.add_argument('--inference-timeout', type=float, default=180.0, help='Inference output timeout, seconds')
    p.add_argument('--boot-timeout', type=float, default=12.0, help='Wait for reset/BOOT READY, seconds')
    p.add_argument('--save-payload', help='Save generated uint8 image/tensor to .bin')
    p.add_argument('--dry-run', action='store_true', help='Create/check payload without connecting to MCU')
    args = p.parse_args(argv)
    try:
        spec = json.loads(Path(args.manifest).read_text()) if args.manifest else {}
        if spec and spec.get('protocol') != PROTOCOL:
            raise ValueError(f"manifest protocol {spec.get('protocol')!r} != {PROTOCOL!r}")
        if spec and spec.get('transport_dtype') != 'uint8':
            raise ValueError('this v1 sender supports raw uint8 transport only')
        shape = parse_shape(args.shape) if args.shape else spec.get('shape', [1, 256])
        n = args.elements if args.elements is not None else numel(shape)
        baud = args.baud or spec.get('baud', 115200)
        chunk = args.chunk or spec.get('chunk_bytes', 32)
        if n < 1 or chunk < 1 or args.repeat < 1:
            raise ValueError('--elements, --chunk and --repeat must be positive')
        data = payload_for(n, args.pattern, args.value, args.seed, args.input_file)
        crc = zlib.crc32(data) & 0xffffffff
        print(f'[info] wire protocol={PROTOCOL} raw_uint8 shape={shape} length={len(data)} '
              f'baud={baud} chunk={chunk} crc32={crc:08X}')
        print('[info] firmware, not this sender, converts raw uint8 pixel to TFLite int8 q=pixel-128')
        if args.save_payload:
            Path(args.save_payload).write_bytes(data)
            print(f'[info] wrote {args.save_payload}')
        if args.dry_run:
            print(f'[dry-run] first 32 bytes: {data[:32].hex(" ")}')
            return 0
        if not args.port:
            p.error('--port is required unless --dry-run is used')
        try:
            import serial  # pyserial
        except ImportError as ex:
            raise RuntimeError('install pyserial: python3 -m pip install pyserial') from ex
        with serial.Serial(args.port, baudrate=baud, timeout=0.1, write_timeout=args.timeout) as ser:
            # Opening Uno USB serial usually toggles reset and prints READY.
            print('[info] waiting for Uno boot READY (port open may reset the board) ...')
            await_token_line(ser, 'READY', args.boot_timeout)
            for rep in range(1, args.repeat + 1):
                print(f'\n[run] {rep}/{args.repeat}')
                cls, ms, output = send_one(ser, data, chunk, args.timeout, args.inference_timeout)
                print(f'[result] class={cls} inference_ms={ms} output_int8={output}')
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError) as ex:
        print(f'[error] {ex}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
