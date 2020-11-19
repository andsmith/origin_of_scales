import numpy as np

PLOT_FONT_SIZES = {'axes.labelsize': 'large',
                   'axes.titlesize':'large',
                   'xtick.labelsize':'small',
                   'ytick.labelsize':'small'}


WAVEFORM_PADDING = 4
LINE_WIDTH = .5

class MultiplotManager(object):
    class __init__(self, shape, dims, font_sizes=None):
        self._shape = shape
        self._finalized = False
        self._font_sizes = font_sizes.copy() if font_sizes is not None else {}

        # start making the figure
        self._fig = plt.figure(figsize=dims)
        self._font_sizes.update(PLOT_FONT_SIZES)
        self._axes.subplots(nrows=shape[0], ncols=shape[1])
        self._fig.rcParams.update(self._font_sizes)



    def save(self, filename, live=False):
        if not self._finalized:
            raise Exception("Plot save before finalized!")
        if live:
            plt.show()
        else:
            plt.savefig(filename)
            print("Wrote:  %s" % (filename, ))

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
    

def make_plot_options_grid(style_grid, changes):
    options = [[BASE_STYLES[style] for style in row] for row in style_grid]

    
def apply_plot_properties(ax, props):

    if 'x_axis_thickness' in props:
        x_lims = ax.get_xlim()
        x_range = np.max((1.0, x_lims[1]- x_lims[0]))
        axis_range = [x_lims[0] - x_range, x_lims[1] + x_range]
        ax.plot(axis_range, [0,0], "-", color='black', linewidth=props['x_axis_thickness'])
        ax.set_xlim(x_lims)

    if 'x_scale' in props:
        ax.set_xscale(props['x_scale'])

    if 'subtitle' in props:
        ax.title.set_text(props['subtitle'])

    if 'xlabel' in props:
        lab = props['xlabel']
        pad = props.get('xlabelpad', 6)
        ax.set_xlabel(lab, labelpad=pad)
        print("Set x-label to: %s" % (lab, ))

    if 'ylabel' in props:
        pad = props.get('ylabelpad', 6)
        lab = props['ylabel']
        ax.set_ylabel(lab, labelpad=pad)
        print("Set y-label to: %s" % (lab, ))

    if 'xlim' in props:
        ax.set_xlim(props['xlim'])

    if 'ylim' in props:
        ax.set_ylim(props['ylim'])

    if 'grid' in props:
        ax.grid(**props['grid'])

    if 'xnoticks' in props:
        ax.set_xtics([])
    if 'ynoticks' in props:
        ax.set_yticks([])
                
    if 'autoscale' in props:
        if 'yscale' in props:
            props['autoscale']['vert'] = False
                
        auto_scale(ax, vert=props['autoscale'].get('vert', False),
                   horiz=props['autoscale'].get('horiz', False),
                   margin=props['autoscale'].get('margin', None)),

    if 'yscale' in props:
        if props['yscale']=='waveform':
            y_lims = ax.get_ylim()
        y_range =  y_lims[1]- y_lims[0]
        axis_range = [y_lims[0] - y_range*WAVEFORM_PADDING, y_lims[1] + y_range*WAVEFORM_PADDING]
        ax.set_ylim(axis_range)


    if 'xnoticklabels' in props:
        ax.set_xticklabels([])
    if 'ynoticklabels' in props:
        ax.set_yticklabels([])
