"""
Module contains the main properties for all plots, based on LaTex document, and basic functions regarding plot
formatting.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# matplotlib.use("Qt5Agg")

# Set font sizes: set according to LaTEX document
SMALL_SIZE = 10
MEDIUM_SIZE = 11
BIGGER_SIZE = 12

# LaTex document properties
textwidth = 448
textheight = 635.5
# plt.rc("text", usetex='True')
plt.rc('font', family='serif', serif='Arial')

# Set plot properties:
# plt.rc('figure', figsize=(15, 15))
plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # font size of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # font size of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # font size of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend font size
plt.rc('figure', titlesize=BIGGER_SIZE)  # font size of the figure title
plt.rc('axes', axisbelow=True)

plt.rc('savefig', dpi=1000)

# Set bar width (histograms)
width = 0.75

colors_scen = ['royalblue', 'limegreen', 'lightcoral', 'gold', 'darkturquoise', 'indigo', 'peru', 'powderblue', 'plum']
markers_scen = ['.', 'x', 'd', '^', '*', 's', 'p', '>', '<']


# Basic functions:
def plot_size(fraction=1, height_fraction=0):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    Source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
        Parameters
        ----------
        :param fraction: float, optional (Fraction of the width which you wish the figure to occupy)
        :param height_fraction: float, optional (Fraction of the entire page height you wish the figure to occupy)

        Returns
        -------
        fig_dim: tuple (Dimensions of figure in inches)
        """
    # Width of figure (in pts)
    fig_width_pt = textwidth * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    if height_fraction == 0 :
        # Figure height in inches: golden ratio
        fig_height_in = fig_width_in * golden_ratio
    else:
        fig_height_in = textheight * inches_per_pt * height_fraction

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim
