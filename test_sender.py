#!/usr/bin/env python3
"""Host-only simulation of the exact Uno IMG/32/K/PRED protocol."""
import importlib.util
from collections import deque
from pathlib import Path

path = Path(__file__).with_name('r15_send_tensor.py')
spec = importlib.util.spec_from_file_location('r15_sender', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class FakeSerial:
    def __init__(self, payload, chunk=32):
        self.expected = payload
        self.queue = deque()
        self.chunk = chunk
        self.stage = 'cmd'
        self.got = bytearray()
        self.pending = bytearray()
        self.blocks = 0
        self.messages = []

    def write(self, buf):
        if self.stage == 'cmd':
            assert buf == b'IMG\n'
            self.messages.append(buf)
            self.queue.extend(b'RDYIMG\r\n')
            self.stage = 'data'
        elif self.stage == 'data':
            self.got.extend(buf)
            self.pending.extend(buf)
            if len(self.pending) == self.chunk:
                self.queue.extend(b'K')
                self.blocks += 1
                self.pending.clear()
            if len(self.got) == len(self.expected):
                assert self.got == self.expected
                self.queue.extend(b'PRED 7 MS 123\r\nOUTPUT_INT8 -128 -128 -128 -128 -128 -128 -128 127 -128 -128\r\nREADY\r\n')
                self.stage = 'done'
        else:
            raise AssertionError('sender wrote after completion')
        return len(buf)

    def read(self, n):
        if self.queue:
            return bytes([self.queue.popleft()])
        return b''

    def flush(self):
        pass

def test():
    data = module.payload_for(256, 'ramp', 0, 1234, None)
    fake = FakeSerial(data)
    cls, ms, out = module.send_one(fake, data, 32, 0.5, 0.5)
    assert (cls, ms, len(out), fake.blocks) == (7, 123, 10, 8)
    assert fake.messages == [b'IMG\n']
    assert fake.got == data
    try:
        module.payload_for(255, 'ramp', 0, 1234, str(Path(__file__).with_name('dummy256.bin')))
    except ValueError:
        pass
    else:
        raise AssertionError('file size mismatch not caught')
    print('PASS: command, 8 acknowledged 32-byte chunks, input bytes, result parsing, size guard')

if __name__ == '__main__':
    test()
