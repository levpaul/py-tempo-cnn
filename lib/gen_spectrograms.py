import os
import librosa
import numpy as np

win_len = 1024

def extract_spectro(f):
    output_dir = os.path.join(os.path.dirname(f), '../specs/')
    if not f.endswith('.wav'):
        print('not a wav file!')
        return

    np.save('{}/{}.npy'.format(output_dir, os.path.basename(f)[:-4]), get_spectro_data(f))


def get_spectro_data(f):
    y, sr = librosa.load(f, sr=11025)
    hop_len = win_len // 2
    data = librosa.feature.melspectrogram(y, sr, n_fft=win_len, hop_length=hop_len, power=1, n_mels=40, fmin=20, fmax=5000)
    return data.astype(np.float16)
