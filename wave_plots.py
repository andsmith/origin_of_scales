import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from util import sine_wave, merge_dict_list, SignalDecomposer
from plot_util import MultiplotManager


def add_wave(manager, row_ind, freq=440.0, t=0.100, res=44100.0, options=None):
    """
    Create two horizontally arranged plots from a pure sine wave, or a weighted mixture of sine waves.
        Left:  time-domain, waveform(s)
        Right:  frequency-domain, FFt, cleaned up
    plotter:  MultiplotManager object
    row_index:  which row of plotter's grid are we drawing to?
    freq: specify just frequency(frequencies) with a float (tuple of floats), or
        more completely with phase, amplitude, color, etc using a dict/list of dicts
        (for options, see "defaults" below).  All but 'freq' are optional for the dict option.
    axes: list of 2 axes, for time domain and freq domain.
    t: time (seconds) to synth
    res:  sample rate

    :returns: pair lists of plots, the ones added to the two axes
    """
    # Set-up:
    options = {} if options is None else options
    time_axis, freq_axis = manager.get_row_axes(row_ind)

    defaults = {'freq': None,
                'amplitude': 1.0,
                'phase_angle': 0.0,
                'spectrum_threshold': 0.05,  # Prune l/r parts of FFT < this fraction of peak power.
                'spectrum_padding': 4,  # After pruning, add back this many on each side (if possible)
                'color': 'black',
                'kwargs': {},  # additional args to matplotlib.plot() command
                'spectrum_kwargs': {},
                }

    waves = [defaults.copy()]
    if isinstance(freq, float):
        waves[0]['freq'] = freq
    elif isinstance(freq, dict):

        waves[0].update(freq)
    elif isinstance(freq, tuple):
        waves = [defaults.copy() for _ in range(len(freq))]
        for i, w in enumerate(waves):
            w['freq'] = freq[i]
    elif isinstance(freq, list):
        waves =[]
        for f in freq:
            waves.append(defaults.copy())
            waves[-1].update(f)

    else:
        raise Exception("Param 'freq' must be float, dict, or list of dicts.")
    return_plots = [[], []]
    cumulative_wave = np.arange(0.0, t, 1.0 / res) * 0

    # Generate the signal(s), plot time domain half
    x_vals = None
    for wave in waves:
        x, y = sine_wave(wave['freq'], wave['amplitude'], wave['phase_angle'], t, res)
        x_vals = x
        cumulative_wave += y
        if 'plot_components' in options:
            raise Exception("Individual component timeseries plots not ipmlemented!")

    return_plots[0].append(time_axis.plot(x_vals, cumulative_wave,
                                          color=waves[0]['color'], **waves[0]['kwargs']))
    # Do signal analysis
    sig = SignalDecomposer(cumulative_wave,
                           res,
                           spectrum_threshold=waves[0]['spectrum_threshold'],
                           spectrum_padding=waves[0]['spectrum_padding'])
    power, freq = sig.get_power()

    # Plot frequency domain half
    spectrum_options = defaults.copy()
    spectrum_options.update(options)
    return_plots[1].append(freq_axis.plot(freq[1:], power[1:],
                                          color=waves[0]['color'], **spectrum_options['spectrum_kwargs']))
    manager.set_row_plots(return_plots, row_ind)
    return return_plots


def auto_scale(axis, vert=False, horiz=False, margin=0.025):
    margin = 0.025 if margin is None else margin

    def widen(interval):
        half_length = (1.0 + margin) * (interval[1] - interval[0]) / 2.0
        center = np.mean(interval)
        return (center - half_length, center + half_length)

    x, y = axis.get_xlim(), axis.get_ylim()
    axis.autoscale_view(tight=True)
    x_tight, y_tight = axis.get_xlim(), axis.get_ylim()
    x_new = widen(x_tight) if horiz is not None and horiz else x
    y_new = widen(y_tight) if vert is not None and vert  else y
    axis.set_xlim(x_new)
    axis.set_ylim(y_new)
    print("Set new limits:  %s, %s" % (x_new, y_new))


def make_figure_1(filename=None, overwrite=False):
    # general size
    w = 4.5  # inches
    h = 3.5
    rows = 3

    # set-up plot styles for each cell
    m = MultiplotManager(cells=[['waveform', 'spectrum']] * rows,
                         dims=(w, h),
                         sharex='col',
                         overwrite=overwrite)

    # customize bottom & top rows with titles & axis labels, etc.
    m.get_cell_style(0, 0).update({'subtitle': 'time domain',})
    m.get_cell_style(0, 1).update({'subtitle': 'frequency domain'})

    m.get_cell_style(-1, 0).update({'xlabel': r"$t$ (sec)",  # Bottom cells get x-ticks & labels
                                    'xticklabels': True,
                                    })

    m.get_cell_style(-1, 1).update({'xlabel': r"$f$ (Hz)",  # spectrum cells get y-tick but no label
                                    'xticklabels': True,
                                    })
    for row_i in range(rows):
        m.get_cell_style(row_i, 1).update({'x_clip': [35.0, None]})

    # Add things:
    f0 = 110.0  # in Hz
    add_wave(m, 0, freq=(f0,))

    amplitudes = np.exp(-np.linspace(3., 5., 7))
    amplitudes /= np.sum(amplitudes)
    freqs = [{'freq': f0*(i+1), 'amplitude': amplitudes[i]} for i in range(len(amplitudes))]

    add_wave(m, 1, freq=freqs)

    freqs = [f0] + (f0 * (1. + 4.23 * np.random.rand(30))).tolist()
    add_wave(m, 2, freq=tuple(freqs))

    # Finalize and save.
    m.save(filename)


if __name__ == "__main__":
    make_figure_1()
