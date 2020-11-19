import numpy as np
import matplotlib.pylab as plt
from matplotlib.lines import Line2D as Line2D
import os
from pylab import rcParams as rcp

WAVEFORM_MARGIN = 0.5
BUBBLE_MARKER_SIZE = 1.5
LINE_WIDTH = 1

BASE_STYLES = {'waveform': {'auto_range': 'wave',
                            'yticks': False,
                            'xticks': False,
                            'yticklabels': False,
                            'linestyle': "-",  # marker/line-style options to matplotlib
                            'linewidth': LINE_WIDTH,
                            'xticklabels': False,},

               'spectrum': {'grid': {'which': 'both', 'axis': 'both'},
                            'yticks': True,
                            'xticks': True,
                            'yticklabels': False,
                            'xticklabels': False,
                            'linestyle': "-",
                            'marker': "o",
                            'markersize': BUBBLE_MARKER_SIZE,
                            'linewidth': LINE_WIDTH,
                            'color': 'black',
                            'ylabelpad': 0,
                            'ylabel': r"power",
                            'xscale': 'log',}}

PLOT_FONT_SIZES = {'axes.labelsize': 'large',
                   'axes.titlesize': 'large',
                   'xtick.labelsize': 'small',
                   'ytick.labelsize': 'small'}


class MultiplotManager(object):
    def __init__(self, cells, dims, font_sizes=None, overwrite=False):
        self._n_rows = len(cells)
        self._n_cols = len(cells[0])

        self._cells = [[BASE_STYLES[c] for c in row] for row in cells]
        self._finalized = False
        self._overwrite = overwrite
        self._font_sizes = font_sizes.copy() if font_sizes is not None else {}

        # start making the figure
        self._fig = plt.figure(figsize=dims)
        self._font_sizes.update(PLOT_FONT_SIZES)
        self._axes = self._fig.subplots(nrows=self._n_rows, ncols=self._n_cols)
        if self._n_rows == 1:  # stoopid matplotlib!
            self._axes = [self._axes]
        rcp.update(self._font_sizes)

    def get_cell_style(self, row, col):
        return self._cells[row][col]

    def get_row_axes(self, row):
        return self._axes[row]

    def _finalize_plot(self):
        """
        Go through each axis, make sure style cells are applied.
        """

        for r, row in enumerate(self._axes):
            for c, axis in enumerate(row):
                apply_plot_properties(axis, self._cells[r][c])
        self._finalized = True

    def save(self, filename):
        self._finalize_plot()
        if not self._finalized:
            raise Exception("Plot save before finalized!")
        if filename is None:
            plt.show()
        else:
            if os.path.exists(filename):
                raise Exception("File already exists!")
            plt.savefig(filename)
            print("Wrote:  %s" % (filename,))


def make_plot_options_grid(style_grid, changes=None):
    return [[BASE_STYLES[style] for style in row] for row in style_grid]


def _get_lines_from_axis(ax):
    return [x for x in ax.get_children() if isinstance(x, Line2D)]


def _get_limits_from_axis(ax):
    line_children = _get_lines_from_axis(ax)
    dx = np.hstack([line.get_xdata() for line in line_children])
    dy = np.hstack([line.get_ydata() for line in line_children])
    xlim, ylim = (np.min(dx), np.max(dx)), (np.min(dy), np.max(dy))
    return xlim, ylim


def apply_plot_properties(ax, props):
    """
    Make sure options are on (call before displaying/saving).
    Order of application is important!
    Unrecognized options throw exception.

    :param ax: Axis object
    :param props: dict with plot options  (see comments in code)
    """
    props = props.copy()  # remove as applied, to check for unhandled

    auto_range = props.pop('auto_range', None)

    if auto_range == 'wave':
        # Annoying way to get to plot data
        xlim, ylim = _get_limits_from_axis(ax)
        y_range = ylim[1] - ylim[0]
        margin = WAVEFORM_MARGIN * y_range / 2.0
        new_xlim = xlim
        new_ylim = (ylim[0] - margin, ylim[1] + margin)
        ax.set_ylim(new_ylim)
        ax.set_xlim(new_xlim)

    elif auto_range is not None:
        raise Exception("auto_range option not recognized:  %s" % (auto_range,))

    if not props.get('xticks', False):
        ax.set_xticks([])
    if not props.get('xticklabels', False):
        ax.set_xticklabels([])
    if not props.get('yticks', False):
        ax.set_yticks([])
    if not props.get('yticklabels', False):
        ax.set_yticklabels([])

    if 'subtitle' in props:
        ax.title.set_text(props['subtitle'])

    if 'xlabel' in props:
        lab = props['xlabel']
        pad = props.get('xlabelpad', 6)
        ax.set_xlabel(lab, labelpad=pad)
        print("Set x-label to: %s" % (lab,))

    if 'ylabel' in props:
        pad = props.get('ylabelpad', 6)
        lab = props['ylabel']
        ax.set_ylabel(lab, labelpad=pad)
        print("Set y-label to: %s" % (lab,))

    if 'grid' in props:
        ax.grid(**props['grid'])

    if 'x_clip' in props:
        xlim, _ = _get_limits_from_axis(ax)
        x_lo = props['x_clip'][0] if props['x_clip'][0] is not None else xlim[0]
        x_hi = props['x_clip'][1] if props['x_clip'][1] is not None else xlim[1]
        new_xlim = [x_lo, x_hi]
        print("Clipping X to:  %s" % (new_xlim,))
        ax.set_xlim(new_xlim)

    name_settable_properties_axis = ['xscale', 'yscale']
    name_settable_properties_line2d = ['marker', 'linestyle', 'markersize', 'linewidth']

    lines = _get_lines_from_axis(ax)
    ns_props = {p: props[p] for p in name_settable_properties_line2d if p in props}
    for line in lines:
        line.set(**ns_props)
    ns_props = {p: props[p] for p in name_settable_properties_axis if p in props}
    ax.set(**ns_props)

    print("Setting:\n\t%s" % ("\n\t".join(["%s: %s" % (p, ns_props[p]) for p in ns_props]),))
