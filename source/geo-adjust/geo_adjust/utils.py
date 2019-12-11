from numpy import sqrt, diag


def compute_corr_coeff(sigma):
    """
    Compute the Pearson correlation coefficient matrix (PCC Matrix) from VCM
    (see also numpy.corrcoef() for direct computation from samples)
    :param sigma: VCM
    :return: PCC Matrix
    """
    d = sqrt(diag(sigma))[None, :]
    pcc = sigma / d / d.T
    return pcc


def lookahead(iterable):
    """Pass through all values from the given iterable, augmented by the
    information if there are more values to come after the current one
    (True), or if it is the last value (False).
    """
    # Get an iterator and pull the first value.
    it = iter(iterable)
    last = next(it)
    # Run the iterator to exhaustion (starting from the second value).
    for val in it:
        # Report the *previous* value (more to come).
        yield last, True
        last = val
    # Report the last value.
    yield last, False


def format_axis(ax):
    # bugfix for sans-serif tick labels (https://stackoverflow.com/a/51409039)
    ax.set_axisbelow(True)
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    ax.grid(linestyle=':')
    ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)


def format_figure(fig=None, share_x=False):
    import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.gcf()

    for x, has_more in lookahead(list(fig.get_axes())):
        if has_more and share_x:
            x.tick_params(labelbottom=False)
        format_axis(x)
    # plt.tight_layout()