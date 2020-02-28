#!/usr/bin/env python


import sox
import uuid

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
# with Pool(40) as p: p.map(my_function, my_array)

pool = ThreadPool(40)

files = [
    'raw/classic.wav',
    'raw/classic.wav',
    'raw/classic.wav',
    'raw/classic.wav',
    'raw/classic.wav',
    'raw/classic.wav',
    'raw/classic.wav',
    'raw/classic.wav',
    'raw/classic.wav',
    'raw/classic.wav',
    'raw/classic.wav',
    'raw/classic.wav',
    'raw/classic.wav'
]

c = 0
def get_file():
    global c
    c += 1
    if c % 100 == 0:
        return
    yield 'raw/classic.wav'


def transform(f):
    tfm = sox.Transformer()
    tfm.lowpass(5500)
    tfm.tempo(.76)
    tfm.echo()
    tfm.flanger()
    tfm.build(get_file, 'out/{}.wav'.format(uuid.uuid4().hex))

pool.map(transform, files)

# Sox Commands I Like:

# chorus gain-in gain-out delay decay speed depth [ -s | -t ] (sin/triangle)
# play classic.wav chorus 1 1 55 .4 .25 10 -t

# Tempo