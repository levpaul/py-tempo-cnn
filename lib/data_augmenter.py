import sox
import random
import glob
import string
import os

class DataAugmenter:
    def __init__(self, source_dir, output_dir, n_times):
        self.files = []
        for f in glob.iglob(source_dir + '/*.wav'):
            self.files.append(f)
        self.n_times = n_times
        self.idx = 0
        self.output_dir = output_dir

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < self.n_times:
            result = self.files[self.idx % len(self.files)]
            self.idx += 1
            return (result, self.output_dir)
        raise StopIteration

# TODO: Pass this in as a JSON/YAML file
effect_probs = {
    'chorus': 0.2,
    'compression': 0.3,
    'delay': 0.3,
    'flanger': 0.2,
    'highpass': 0.1,
    'lowpass': 0.1,
    'overdrive': 0.2,
    'phaser': 0.2,
    'reverb': 0.3,
    'tremolo': 0.2,
}
#
# effect_probs = {
#     'chorus': 0.0,
#     'compression': 0.0,
#     'delay': 0.0,
#     'flanger': 0.0,
#     'highpass': 0.0,
#     'lowpass': 0.0,
#     'overdrive': 0.0,
#     'phaser': 0.0,
#     'reverb': 0.0,
#     'tremolo': 0.0,
# }
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
    tfm.lowpass(frequency=random.randint(3000,10000))

def overdrive_effect(tfm):
    od_gain = random.randint(5,30)
    od_color = random.randint(10,30)
    tfm.overdrive(gain_db=od_gain, colour=od_color)
    tfm.gain(limiter=True)

def phaser_effect(tfm):
    ph_delay = random.randint(1,5)
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

def apply_effect(tfm, effect_func, chance, file_meta):
    if random.random() < chance:
        file_meta['name'] += '-'+effect_func.__name__[:2]
        effect_func(tfm)
        tfm.gain(normalize=True)

# Files from DB
def transform(f, output_dir):
    tfm = sox.Transformer()

    # Get new tempo
    new_tempo = random.randint(50,150)
    orig_tempo = 100.
    tempo_factor = new_tempo/orig_tempo

    # Create new random cut based on new tempo and save
    window_len = 11.9 + 0.1
    scaled_win = window_len * tempo_factor
    orig_len = sox.file_info.duration(f)
    rand_start = random.random() * (orig_len-scaled_win)
    rand_end = rand_start + scaled_win
    tfm.trim(rand_start, rand_end)

    pitch_change = random.randint(-12,12)
    tfm.pitch(n_semitones=pitch_change)
    tfm.gain(normalize=True)

    old_file_short = '-'.join(f[:-4].split('-')[2:4])
    rand_seq = ''.join(random.choices(string.ascii_uppercase + string.digits, k=9))
    file_meta = {'name': '{:s}-{:s}-pi{:+d}'.format(rand_seq, old_file_short,pitch_change)}

    apply_effect(tfm, compression_effect, effect_probs['compression'], file_meta)
    apply_effect(tfm, chorus_effect, effect_probs['chorus'], file_meta)
    apply_effect(tfm, delay_effect, effect_probs['delay'], file_meta)
    apply_effect(tfm, flanger_effect, effect_probs['flanger'], file_meta)
    apply_effect(tfm, highpass_effect, effect_probs['highpass'], file_meta)
    apply_effect(tfm, lowpass_effect, effect_probs['lowpass'], file_meta)
    apply_effect(tfm, overdrive_effect, effect_probs['overdrive'], file_meta)
    apply_effect(tfm, phaser_effect, effect_probs['phaser'], file_meta)
    apply_effect(tfm, reverb_effect, effect_probs['reverb'], file_meta)
    if not '-re' in file_meta['name']:  # TODO: Reverb and tremolo BOTH in a sox chain cuts the lenght in half??
        apply_effect(tfm, tremolo_effect, effect_probs['tremolo'], file_meta)
    tfm.lowpass(5500)

    if abs(tempo_factor - 1.0) < 0.1:
        tfm.stretch(1/tempo_factor)
    else:
        tfm.tempo(tempo_factor, audio_type='m')
    tfm.gain(normalize=True, limiter=True)

    file_meta['name'] += '-{:d}bpm'.format(new_tempo)
    final_filename = '{}/{}.wav'.format(output_dir, file_meta['name'])
    tfm.build(f, final_filename, return_output=True)
