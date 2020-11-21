import numpy as np


def sine_wave(f, a, w, t, res=44100):
    x = np.arange(0.0, t, 1.0 / res)
    y = np.sin((x + w) * 2.0 * np.pi * f) * a

    return x, y


def merge_dict_list(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i].copy())
        z[-1].update(y[i])
    return z


class SignalDecomposer(object):
    def __init__(self, x, rate=44100.0, spectrum_threshold=.2, spectrum_padding=5, zero_upper=True):
        self.x = x
        self.rate = rate
        self.fft = np.fft.fft(x)
        self.mag = 2.0 * np.sqrt(self.fft.real ** 2.0 + self.fft.imag ** 2.0)
        self.dc = self.mag[0]
        if zero_upper:
            self.mag[int(self.mag.size / 2):] = np.NaN

        self.freq_bins = rate * np.linspace(0.0, x.size - 1, x.size) / x.size
        self._thresh = spectrum_threshold
        self._pad = spectrum_padding

    def get_power(self, no_crop=True):
        spec_margin = .5
        min_low_to_crop = 10

        p = self.mag
        freqs = self.freq_bins
        if self._thresh is not None or not no_crop:
            # min_val = np.exp((np.log(np.nanmax(self.mag))*self._thresh + (1.0 - self._thresh)*np.log(np.nanmin(self.mag))))
            min_val = np.nanmax(self.mag) * self._thresh

            # on the left, trim where power < threshold
            lowest_usable_index = np.min(np.where(self.mag >= min_val))
            lowest_usable_index = np.max((lowest_usable_index - 2, 0))
            # unless near the beginning
            if lowest_usable_index < min_low_to_crop:
                lowest_usable_index = 0

            # on the right, trim to 10% after final decreasing monotonicity begins
            first_half = int(self.mag.size/2)-1

            increase_indices = np.where(np.diff(self.mag[:first_half]) > 0.0)[0]

            if increase_indices.size > 0:
                final_increase = np.max(increase_indices)
                margin = int(spec_margin * (final_increase - lowest_usable_index))
                highest_usable_index = final_increase + margin
                highest_usable_index = np.min((highest_usable_index - 1, first_half-1))
                print(final_increase,margin, highest_usable_index)
            else:
                highest_usable_index = first_half

            print("Pruning to %i bins." % (highest_usable_index,))
            p = p[lowest_usable_index:highest_usable_index]
            freqs = freqs[lowest_usable_index:highest_usable_index]
        return p, freqs
