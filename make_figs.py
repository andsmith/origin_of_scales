import argparse
from wave_plots import make_figure_1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate figures.')
    parser.add_argument('--height', '-t', help="height of figure in inches", type=float, default=2.0)
    parser.add_argument('--width', '-w', help="width of figure in inches", type=float, default=4.5)
    parser.add_argument('--plot', '-p', help="plot to screen instead of file", action='store_true', default=False)
    parser.add_argument('--clobber', '-c', help="overwrite existing files", action='store_true', default=False)
    parsed = parser.parse_args()
    print("Generating figure(s) with:\n\twidth:  %s\n\theight:  %s" % (parsed.width, parsed.height))

    # Generate all figures:

    make_figure_1(filename="figure_1.pgf", overwrite=parsed.clobber)
