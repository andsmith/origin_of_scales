import pprint
import numpy as np
import matplotlib.pylab as plt
from matplotlib.lines import Line2D as Line2D
import os
from pylab import rcParams as rcp

WAVEFORM_MARGIN = 0.5
BUBBLE_MARKER_SIZE = 1.2
LINE_WIDTH = .5

BASE_STYLES = {'waveform': {'auto_range': 'wave',
                            'yticks': False,
                            'xticks': True,
                            'xticklabels': False,
                            'yticklabels': False,
                            'linestyle': "-",  # marker/line-style options to matplotlib
                            'linewidth': LINE_WIDTH,},

               'spectrum': {'grid': {'which': 'both', 'axis': 'both'},
                            'yticks': True,
                            'xticks': True,
                            'yticklabels': False,
                            'xticklabels': False,
                            'xminor': True,
                            'linestyle': "-",
                            'marker': "o",
                            'markersize': BUBBLE_MARKER_SIZE,
                            'linewidth': LINE_WIDTH,
                            'color': 'black',
                            'ylabelpad': 0,
                            'xscale': 'log'}}

PLOT_FONT_SIZES = {'axes.labelsize': 'large',
                   'axes.titlesize': 'large',
                   'xtick.labelsize': 'small',
                   'ytick.labelsize': 'small'}


class MultiplotManager(object):
    def __init__(self, cells, dims, font_sizes=None, overwrite=False, **subplot_kwargs):
        self._n_rows = len(cells)
        self._n_cols = len(cells[0])

        self._cells = [[BASE_STYLES[c].copy() for c in row] for row in cells]
        self._plots = [None for _ in range(self._n_rows)]
        self._finalized = False
        self._overwrite = overwrite
        self._font_sizes = font_sizes.copy() if font_sizes is not None else {}

        # start making the figure
        self._fig = plt.figure(figsize=dims)
        self._font_sizes.update(PLOT_FONT_SIZES)
        self._axes = self._fig.subplots(nrows=self._n_rows, ncols=self._n_cols, **subplot_kwargs)
        if self._n_rows == 1:  # stoopid matplotlib!
            self._axes = [self._axes]
        rcp.update(self._font_sizes)

    def get_cell_style(self, row, col):
        return self._cells[row][col]

    def get_row_axes(self, row):
        return self._axes[row]

    def set_row_plots(self, plots, index):
        self._plots[index] = plots

    def _finalize_plot(self):
        """
        Go through each axis, make sure style cells are applied.
        """
        for r, row in enumerate(self._axes):
            for c, axis in enumerate(row):
                apply_plot_properties(axis, self._plots[r][c], self._cells[r][c], "plot (%i, %i)-" % (r, c))
        self._finalized = True

    def save(self, filename):
        self.print_axis_state()
        self._finalize_plot()
        self.print_axis_state()
        # if not self._finalized:
        #    raise Exception("Plot save before finalized!")
        if filename is None:
            plt.show()
        else:
            if os.path.exists(filename):
                raise Exception("File already exists!")
            plt.savefig(filename)
            print("Wrote:  %s" % (filename,))

    def print_axis_state(self):

        def extract_and_print(p):
            v = []

            for r in range(len(self._axes)):
                v.append([])
                for c in range(len(self._axes[r])):
                    val = len(self._axes[r][c].properties()[p])
                    v[-1].append(val)

            print("%s - " % (p, ))
            pprint.pprint(v)
            print("")
        extract_and_print('xticklabels')
        extract_and_print('yticklabels')


def make_plot_options_grid(style_grid, changes=None):
    return [[BASE_STYLES[style] for style in row] for row in style_grid]


def _get_lines_from_plots(plots):
    l = [line for lines in plots for line in lines]
    return l


def _get_limits_from_plots(plots):
    lines = _get_lines_from_plots(plots)
    dx = np.hstack([line.get_xdata() for line in lines])
    dy = np.hstack([line.get_ydata() for line in lines])
    xlim, ylim = (np.min(dx), np.max(dx)), (np.min(dy), np.max(dy))
    return xlim, ylim


def _handle_axis_appearance(ax, plots, props, verbs):
    # figure our state vars
    x_dir = 'in' if 'xticklabels' in props and 'xticks' in props and not props['xticklabels'] else 'out'
    y_dir = 'in' if 'yticklabels' in props and 'yticks' in props and not props['yticklabels'] else 'out'
    x_ticks = props.get('xticks', False)
    y_ticks = props.get('xticks', False)
    x_ticklabels = props.get('xticklabels', False)
    y_ticklabels = props.get('yticklabels', False)
    x_minor = props.get('xminor', False)
    y_minor = props.get('yminor', False)

    print("%s - tick state:  X=(T-%s, L-%s, M-%s, %s), Y=(T-%s, L-%s, M-%s, %s)" % (
        verbs, x_ticks, x_ticklabels, x_dir, x_minor,
        y_ticks, y_ticklabels, y_dir, y_minor))

    ax.tick_params('x', direction=x_dir, bottom=x_ticks, labelbottom=x_ticklabels)
    ax.tick_params('x', which='minor', bottom=x_minor)
    ax.tick_params('y', direction=y_dir, left=y_ticks, labelleft=y_ticklabels)
    ax.tick_params('y', which='minor', bottom=y_minor)

    if 'subtitle' in props:
        ax.title.set_text(props['subtitle'])

    if 'xlabel' in props:
        lab = props['xlabel']
        pad = props.get('xlabelpad', 6)
        ax.set_xlabel(lab, labelpad=pad)
        #print("Set x-label to: %s" % (lab,))

    if 'ylabel' in props:
        pad = props.get('ylabelpad', 6)
        lab = props['ylabel']
        ax.set_ylabel(lab, labelpad=pad)
        #print("Set y-label to: %s" % (lab,))

    if 'grid' in props:
        ax.grid(**props['grid'])

    return props

def apply_plot_properties(ax, plots, props, verbs):
    """
    Make sure options are on (call before displaying/saving).
    Order of application is important!
    Unrecognized options throw exception.

    :param ax: Axis object
    :param props: dict with plot options  (see comments in code)
    """
    props = props.copy()  # remove as applied, to check for unhandled
    # Axis & lick mark labels
    props = _handle_axis_appearance(ax, plots, props, verbs)

    # viewing range
    auto_range = props.pop('auto_range', None)
    if auto_range == 'wave':
        xlim, ylim = _get_limits_from_plots(plots)
        y_range = ylim[1] - ylim[0]
        margin = WAVEFORM_MARGIN * y_range / 2.0
        new_xlim = xlim
        new_ylim = (ylim[0] - margin, ylim[1] + margin)
        ax.set_ylim(new_ylim)
        ax.set_xlim(new_xlim)
    elif auto_range is not None:
        raise Exception("auto_range option not recognized:  %s" % (auto_range,))
    if 'x_clip' in props:
        xlim, _ = _get_limits_from_plots(plots)
        x_lo = props['x_clip'][0] if props['x_clip'][0] is not None else xlim[0]
        x_hi = props['x_clip'][1] if props['x_clip'][1] is not None else xlim[1]
        new_xlim = [x_lo, x_hi]
        #print("Clipping X to:  %s  (was %s)" % (new_xlim, xlim))
        ax.set_xlim(new_xlim)

    # everything else
    name_settable_properties_axis = ['xscale', 'yscale']
    name_settable_properties_line2d = ['marker', 'linestyle', 'markersize', 'linewidth']
    lines = _get_lines_from_plots(plots)
    ns_props = {p: props[p] for p in name_settable_properties_line2d if p in props}
    for line in lines:
        line.set(**ns_props)
    ns_props = {p: props[p] for p in name_settable_properties_axis if p in props}
    ax.set(**ns_props)
