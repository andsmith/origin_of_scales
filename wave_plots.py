import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from util import sine_wave, merge_dict_list, SignalDecomposer
from plot_util import MultiplotManager, WAVEFORM_PADDING, LINE_WIDTH



def plot_waves(waves, axes, t=0.100, res = 44100.0, fft=True, extras = {}):
    """
    waves:  list of dicts, each with keys in {'freq', 'amp', 'color', 'plot_string', 'phase_angle'}, all but first are optional.
    axes: list of 2 axes, for time domain and freq domain.
    t: time (seconds) to synth
    res:  sample rate
    options:  dict with options:
        title: text, [None]
        x_label: text, [None]
        y_label: text, [None]
    """

    import pprint
    pprint.pprint(self._tones)
    time_axis = axes[0]

    plots = []
    y_values = []
    transforms = []
    y_sum = None
    for tone in self._tones:
        tone = tone.copy()
        plot_string = tone.pop('plot_string', '-')
        color = tone.pop('color', 'black')
        freq = tone.pop('freq')
        amp = tone.pop('amp', 1.0)
        phase = tone.pop('phase_angle',0.0)
        x, y = sine_wave(freq, amp, phase, t, res)
        y_values.append(y)
        y_sum = y_sum+y if y_sum is not None else y
        if 'plot_components' in extras[0]:
            plots.append(time_axis.plot(x, y, plot_string, color=color, **tone))

    plots.append(time_axis.plot(x, y_sum, plot_string, color=color, **tone))


    apply_plot_properties(time_axis, extras[0])

    freq_axis = axes[1]
    sig = SignalDecomposer(y_sum, res)
    color = tone.pop('color', 'black')
    thresh = extras[1].get('prune', {'thresh':None})['thresh']
    margin = extras[1].get('prune', {'margin':None})['margin']
    power, freq = sig.get_power(prune_threshold =thresh, prune_margin = margin)
    plot_str = extras[1].get('plot_str', "-")
    print(freq[1:].min(), freq[1:].max(), power[1:].min(), power[1:].max())
    freq_axis.plot(freq[1:], power[1:], plot_str, color=color, **extras[1]['plot_args'])

    apply_plot_properties(freq_axis, extras[1])

    if 'prune' in extras[1] and 'min_freq_hz' in extras[1]['prune']:
        xlim = freq_axis.get_xlim()
        xlim = [np.max([extras[1]['prune']['min_freq_hz'], xlim[0]]), xlim[1]]
        freq_axis.autoscale(enable=False, axis='x', tight=True)
        freq_axis.set_xlim(xlim)
        freq_axis.set_xticklabels([])

    return plots


def auto_scale(axis, vert=False, horiz=False, margin = 0.025):
    margin = 0.025 if margin is None else margin
    def widen(interval):
        half_length = (1.0 + margin) * (interval[1]-interval[0]) / 2.0
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



def make_figure_1(w, h, rows=3, cols=2, overwrite=False):

    # set-up
    m = MultiplotManager(shape=(rows, cols), dims=(w,h), overwrite = True)

    # add things
    f0 = 111.1111
    add_wave(m, freq=f0)

    top_params = [{'subtitle':'time domain', 'yscale': 'waveform'},
                  {'subtitle':'frequency domain'}]

    bottom_params = [{'xlabel': r"$t$ (sec)", 'yscale': 'waveform'},
                     {'xlabel': r"$f$ (Hz)"}]

    middle_params = [{'xnoticklabels': True, 'yscale': 'waveform'}, 
                     {'xnoticklabels': True}]

    common_params = [{'ynoticks':True, 'yscale': 'waveform'},
                     {'plot_str': "o-",
                      'ylabelpad': -2,'ylabel': r"power",
                      'ynoticklabels': True,
                      'x_scale': 'log',
                      'grid': {'which': 'both', 'axis':'both'},
                      'autoscale': {'vert': True, 'horiz': True},
                      'prune':{'thresh': 0.5, 'margin': 5, 'min_freq_hz': 50.0} ,
                      'plot_args':{'linewidth':LINE_WIDTH, 'markersize': 1.2}}]

    top_extras = merge_dict_list(common_params, top_params)
    middle_extras = merge_dict_list(common_params, middle_params)
    bottom_extras = merge_dict_list(common_params, bottom_params)
    
    # make figure
    print(1)
    wp.plot_wave(axes=axes[0], extras=top_extras) 
                                      
    if True:
        wp.add_tone(freq= f0* 2, amp=1./2, linewidth=LINE_WIDTH)
                                      
        print(2)
        wp.plot_wave(axes=axes[1], extras=middle_extras)
        wp.add_tone(freq= f0* 3, amp=1./3, linewidth=LINE_WIDTH)
        print(3)
        wp.plot_wave(axes=axes[2], extras=bottom_extras)
                                      


if __name__=="__main__":
    make_figure_1(w=4.5, h=3)
