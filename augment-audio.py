#!/usr/bin/env python


import sox
import uuid
import random

from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat

# No diff between poolings except normal Pool gives Exceptions at runtime
pool = ThreadPool(50)
files = repeat('raw/classic.wav', times=10)

effect_probs = {
    "chorus": 0.2,
    "compression": 0.35,
    "delay": 0.4,
    "flanger": 0.4,
    "highpass": 0.1,
    "lowpass": 0.1,
    "overdrive": 0.45,
    "phaser": 0.3,
    "reverb": 0.55,
    "tremolo": 0.25,
}
# ============================================================
#      EFFECTS
# ============================================================

def chorus_effect(tfm):
    voices = random.randint(2,5)
    chr_shapes = ['s' if random.random() < 0.5 else 't' for _ in range(voices)]
    chr_delays = [random.randint(20,100) for _ in range(voices)]
    chr_speeds = [random.randint(10,499)/100. for _ in range(voices)]
    tfm.chorus(n_voices=voices, shapes=chr_shapes, delays=chr_delays, speeds=chr_speeds)

def compression_effect(tfm):
    tfm.contrast(amount=random.randint(50,100))

def delay_effect(tfm):
    echos = random.randint(1,5)
    dl_delays = [random.randint(14,800) for _ in range(echos)]
    dl_decays = [random.randint(20,60)/100. for _ in range(echos)]
    tfm.echo(n_echos=echos, delays=dl_delays, decays=dl_decays)

def flanger_effect(tfm):
    fl_depth = random.randint(2,10)
    fl_speed = random.randint(10,1000)/100.
    fl_shape = 'sine' if random.random() < 0.5 else 'triangle'
    tfm.flanger(depth=fl_depth, speed=fl_speed, shape=fl_shape)

def highpass_effect(tfm):
    tfm.highpass(frequency=random.randint(30,500))

def lowpass_effect(tfm):
    tfm.lowpass(frequency=random.randint(3000,15000))

def overdrive_effect(tfm):
    od_gain = random.randint(5,30)
    od_color = random.randint(10,30)
    tfm.overdrive(gain_db=od_gain, colour=od_color)
    tfm.gain(limiter=True)

def phaser_effect(tfm):
    ph_delay = random.randint(0,5)
    ph_decay = random.randint(10,50)/100.
    ph_speed = random.randint(10,200)/100.
    ph_shape = 'sinusoidal' if random.random() < 0.5 else 'triangular'
    tfm.phaser(delay=ph_delay, decay=ph_decay, speed=ph_speed, modulation_shape=ph_shape)

def reverb_effect(tfm):
    rv_reverb = random.randint(5,70)
    rv_pre_delay = random.randint(0,20)
    tfm.reverb(reverberance=rv_reverb, pre_delay=rv_pre_delay)

def tremolo_effect(tfm):
    tr_speed = random.randint(1,30)
    tr_depth = random.randint(10,60)
    tfm.tremolo(speed=tr_speed, depth=tr_depth)

# ============================================================

def apply_effect(tfm, effect_func, chance):
    if random.random() < chance:
        effect_func(tfm)
        tfm.gain(normalize=True)

# Files from DB
def transform(f):
    tfm = sox.Transformer()

    tfm.pitch(n_semitones=(random.randint(-12,12)))
    tfm.gain(normalize=True)

    apply_effect(tfm, compression_effect, effect_probs["compression"])
    apply_effect(tfm, chorus_effect, effect_probs["chorus"])
    apply_effect(tfm, delay_effect, effect_probs["delay"])
    apply_effect(tfm, flanger_effect, effect_probs["flanger"])
    apply_effect(tfm, highpass_effect, effect_probs["highpass"])
    apply_effect(tfm, lowpass_effect, effect_probs["lowpass"])
    apply_effect(tfm, overdrive_effect, effect_probs["overdrive"])
    apply_effect(tfm, phaser_effect, effect_probs["phaser"])
    apply_effect(tfm, reverb_effect, effect_probs["reverb"])
    apply_effect(tfm, tremolo_effect, effect_probs["tremolo"])
    tfm.lowpass(5500)

    orig_tempo = 100.
    new_tempo = random.randint(50,150)
    tfm.tempo(new_tempo/orig_tempo)

    tfm.gain(normalize=True)

    tfm.build(f, 'out/{}.mp3'.format(uuid.uuid4().hex))

pool.map(transform, files)

# Sox Commands I Like:

# chorus gain-in gain-out delay decay speed depth [ -s | -t ] (sin/triangle)
# play classic.wav chorus 1 1 55 .4 .25 10 -t

# Tempo