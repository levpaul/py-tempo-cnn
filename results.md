### Results

#####Tempo only Dataset 100k
- mfmod1 lr01-big.out ep300=(99.95,16.69)
- mfmod3 lr01-big, ep50=(88.58,33.11)


#####Effects 100k (big-effects)*
- mfmod1, bsize256, lr01, ep##=(,)
- mfmod3, bsize192, lr01, ep##=(,)
- mfmod4, bsize32, lr01, ep##=(,)


#####Effects 500k (huge-effects)*
- mfmod1, bsize256, lr01, ep##=(,)
- mfmod3, bsize192, lr01, ep##=(,)
- mfmod4, bsize32, lr01, ep##=(,)


\* Effects probs:

```
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
```
