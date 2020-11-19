import numpy as np

def sine_wave(f, a, w, t, res=44100):
    x = np.arange(0.0, t, 1.0 / res)
    y = np.sin((x+w)* 2.0 * np.pi  * f) * a
    
    return x, y

def merge_dict_list(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i].copy())
        z[-1].update(y[i])
    return z



class SignalDecomposer(object):
    def __init__(self, x, rate=44100.0):
        self.x = x
        self.rate = rate
        self.fft = np.fft.fft(x)
        self.mag =  2.0* np.sqrt(self.fft.real**2.0 + self.fft.imag**2.0)
        self.dc = self.mag[0]
        self.freq_bins = np.linspace(0.0,rate-1, rate)  * rate/ x.size

    
    def get_power(self, prune_threshold = None, prune_margin = 5):
        p = self.mag
        freqs = self.freq_bins
        if prune_threshold is not None:
            min_val = np.exp((np.log(self.mag.max())*prune_threshold - (1.0 - prune_threshold)*np.log(self.mag.min())))
            lowest_usable_index = np.argmin(np.where(self.mag >= min_val))
            lowest_usable_index = np.max((lowest_usable_index - 2, 0))
            highest_usable_index = np.argmax(np.where(self.mag >= min_val))
            highest_usable_index = np.min((highest_usable_index + prune_margin, self.mag.size))
            print("Pruning to %i bins." % (highest_usable_index, ))
            p = p[lowest_usable_index:highest_usable_index]
            freqs = freqs[lowest_usable_index:highest_usable_index]
        return p, freqs
