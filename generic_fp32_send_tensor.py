#!/usr/bin/env python3
"""Generic FP32 RGB input sender for Noodle Studio's ESP32-S3 Serial IMG v1 firmware.

The exported input_spec.json owns the tensor shape, dtype, byte count and transport.
Serial payloads are raw tensor bytes; quantization and preprocessing remain in the model.
"""
import argparse
import json
import random
import re
import sys
import time
import zlib
from pathlib import Path

PROTOCOL = 'noodle-serial-img-v1'
DEFAULT_SPEC = Path(__file__).resolve().with_name('input_spec.json')


def make_payload(spec, args):
    import math
    import struct
    shape = spec['shape']
    count = math.prod(shape)
    size = spec['bytes']
    if args.input_file:
        data = Path(args.input_file).read_bytes()
    elif args.image:
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise RuntimeError('Install Pillow for --image') from exc
        if len(shape) != 4 or shape[0] != 1 or shape[-1] not in (1, 3):
            raise ValueError('--image supports batch-one grayscale or RGB NHWC input')
        height, width, channels = shape[1:]
        with Image.open(args.image) as original:
            im = ImageOps.exif_transpose(original).convert('L' if channels == 1 else 'RGB')
            im = ImageOps.fit(im, (width, height), method=Image.Resampling.LANCZOS)
            values = [float(v) * args.image_scale for v in im.tobytes()]
        data = struct.pack('<' + str(count) + 'f', *values)
    else:
        if args.pattern == 'zeros':
            values = [0.0] * count
        elif args.pattern == 'constant':
            values = [float(args.value)] * count
        elif args.pattern == 'ramp':
            values = [float(i % 256) / 255.0 for i in range(count)]
        elif args.pattern == 'random':
            rng = random.Random(args.seed)
            values = [rng.random() for _ in range(count)]
        elif args.pattern == 'gradient':
            values = [float(i % 256) / 255.0 for i in range(count)]
        else:
            raise ValueError('Unsupported input pattern')
        data = struct.pack('<' + str(count) + 'f', *values)
    if len(data) != size:
        raise ValueError(f'Input size {len(data)} bytes; model expects exactly {size}')
    return data

def read_line(ser, deadline, max_len=80000):
    line = bytearray()
    while time.monotonic() < deadline:
        b = ser.read(1)
        if not b:
            continue
        if b == b'\n':
            return line.decode('ascii', 'replace').strip()
        if b != b'\r':
            line.extend(b)
            if len(line) > max_len:
                raise RuntimeError('Oversized serial output line; check protocol/baud')
    raise TimeoutError('Timed out reading board response')


def wait_for(ser, token, seconds, log=True):
    limit = time.monotonic() + seconds
    while time.monotonic() < limit:
        line = read_line(ser, limit)
        if log and (not line.startswith('OUTPUT_FLOAT32') or token == 'OUTPUT_FLOAT32'):
            print('[board]', line[:150], flush=True)
        if line == token:
            return
        if line.startswith(('ERR_', 'BOOT_FAILED')):
            raise RuntimeError(line)
    raise TimeoutError(f'No {token} received')


def await_ack(ser, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        c = ser.read(1)
        if c == b'K':
            return
        if c and c not in (b'\r', b'\n'):
            raise RuntimeError(f'Expected K block ACK; received {c!r}')
    raise TimeoutError('No K block ACK from firmware')


def run_inference(ser, data, spec, timeout, inference_timeout, print_output):
    ser.write(b'IMG\n')
    ser.flush()
    wait_for(ser, 'RDYIMG', timeout)
    tick = time.monotonic()
    block_size = spec['chunk_bytes']
    for offset in range(0, len(data), block_size):
        block = data[offset:offset + block_size]
        ser.write(block)
        ser.flush()
        if len(block) == block_size:
            await_ack(ser, timeout)
    print(f'[tx] {len(data)} bytes sent in {time.monotonic()-tick:.3f}s', flush=True)
    predicted = None
    elapsed_ms = None
    out = None
    deadline = time.monotonic() + inference_timeout
    while time.monotonic() < deadline:
        line = read_line(ser, deadline)
        if line.startswith('PRED '):
            match = re.fullmatch(r'PRED\s+(\d+)\s+MS\s+(\d+)', line)
            if not match:
                raise RuntimeError(f'Unexpected PRED format: {line}')
            predicted, elapsed_ms = map(int, match.groups())
            print(f'[result] class={predicted} inference_ms={elapsed_ms}')
        elif line.startswith('OUTPUT_FLOAT32 '):
            out = [float(x) for x in line.split()[1:]]
            if len(out) != spec['output_elements']:
                raise RuntimeError(f'Output length mismatch: {len(out)}')
            if print_output:
                print(line)
            else:
                print(f'[result] output_float32 len={len(out)} min={min(out):.6g} max={max(out):.6g}')
        elif line == 'READY':
            if predicted is None or out is None:
                raise RuntimeError('MCU returned READY without a complete prediction/output')
            return predicted, elapsed_ms, out
        elif line.startswith(('ERR_', 'BOOT_FAILED')):
            raise RuntimeError(line)
        elif line and not line.startswith('OP '):
            print('[board]', line[:150])
    raise TimeoutError('Timed out waiting for inference completion')


def main(argv=None):
    p = argparse.ArgumentParser(description='Send raw little-endian FP32 model input to Noodle on ESP32-S3')
    p.add_argument('--manifest', default=str(DEFAULT_SPEC), help='Auto-generated input_spec.json')
    p.add_argument('--port', help='Serial port, e.g. /dev/ttyACM0; required except --dry-run')
    p.add_argument('--baud', type=int, help='Override baud rate in manifest')
    p.add_argument('--input-file', help='Raw input tensor file; must match exact byte count')
    p.add_argument('--image', help='Grayscale or RGB image, resized to input tensor (requires Pillow)')
    p.add_argument('--image-scale', type=float, default=1.0/255.0, help='Float multiplier for pixel byte values (default 1/255)')
    p.add_argument('--pattern', choices=['gradient', 'zeros', 'ramp', 'constant', 'random'], default='zeros')
    p.add_argument('--value', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--timeout', type=float, default=10.0, help='Per-block ACK and start timeout (seconds)')
    p.add_argument('--boot-timeout', type=float, default=30.0)
    p.add_argument('--inference-timeout', type=float, default=600.0)
    p.add_argument('--dump-output', action='store_true')
    p.add_argument('--save-payload', help='Save actual raw input bytes for TFLite comparison')
    p.add_argument('--dry-run', action='store_true', help='Validate/build payload without hardware')
    args = p.parse_args(argv)
    try:
        spec = json.loads(Path(args.manifest).read_text())
        if spec['protocol'] != PROTOCOL or spec['transport_dtype'] != 'float32-le':
            raise ValueError('Unsupported serial manifest/protocol')
        if spec['dtype'] != 'FLOAT32' or spec['bytes'] != 4*__import__('math').prod(spec['shape']):
            raise ValueError('Inconsistent FP32 transport input specification')
        if args.image and args.input_file:
            raise ValueError('Choose --image or --input-file, not both')
        data = make_payload(spec, args)
        crc = zlib.crc32(data) & 0xffffffff
        print(f'[input] shape={spec["shape"]} dtype=FLOAT32 transport=FLOAT32_LE bytes={len(data)} crc32={crc:08X} '
              f'first16={data[:16].hex(" ")}')
        if args.save_payload:
            Path(args.save_payload).write_bytes(data)
            print(f'[input] saved {args.save_payload}')
        if args.dry_run:
            return 0
        if not args.port:
            p.error('--port is required unless --dry-run')
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError('Install pyserial: python3 -m pip install pyserial') from exc
        with serial.Serial(args.port, args.baud or spec['baud'], timeout=0.1, write_timeout=args.timeout) as ser:
            # HELLO works even if opening the USB CDC port does not reset the MCU.
            time.sleep(0.4)
            ser.write(b'HELLO\n')
            ser.flush()
            wait_for(ser, 'READY', args.boot_timeout)
            for run in range(args.repeat):
                print(f'[run] {run+1}/{args.repeat}')
                run_inference(ser, data, spec, args.timeout, args.inference_timeout, args.dump_output)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, TimeoutError) as exc:
        print(f'[error] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
