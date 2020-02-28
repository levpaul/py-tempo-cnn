#!/usr/bin/env python


import sox
import uuid

from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
# with Pool(40) as p: p.map(my_function, my_array)
from itertools import repeat

# No diff between poolings except normal Pool gives Exceptions at runtime
pool = ThreadPool(50)
# pool = Pool(50)
files = repeat('raw/classic.wav', times=100000)

def transform(f):
    tfm = sox.Transformer()
    tfm.lowpass(5500)
    tfm.tempo(.76)
    tfm.echo()
    tfm.flanger()
    tfm.build(f, 'out/{}.wav'.format(uuid.uuid4().hex))

pool.map(transform, files)

# Sox Commands I Like:

# chorus gain-in gain-out delay decay speed depth [ -s | -t ] (sin/triangle)
# play classic.wav chorus 1 1 55 .4 .25 10 -t

# Tempo