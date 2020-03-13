import glob
import numpy as np

def squash_spectros(folder, output_filename, n):
    data = np.empty((n,1,40,256), dtype=np.float16)
    labels = np.empty(n, dtype=int)
    count = 0
    for f in glob.iglob(folder + '/*.npy', recursive=False):
        if count % 10000 == 0:
            print('Current count',count)
        n = np.load(f)
        data[count,0] = n[:,:256]
        labels[count] = int(f[:-7].split('-')[-1]) # Extract bpm from filename
        count += 1

    print('data dims: ', data.shape)
    print('label dims: ', labels.shape)

    np.save(output_filename + '-data.npy',data)
    np.save(output_filename + '-labels.npy',labels)
