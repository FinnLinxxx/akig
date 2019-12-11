import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML


def show_matrix(m):
    with pd.option_context('display.float_format', lambda x: "%g" % x):
        display(pd.DataFrame(m))


def show_colored_matrix(m):
    def background_gradient(s, m, M, cmap='RdBu', low=0, high=0):
        from matplotlib import colors
        rng = M - m
        norm = colors.Normalize(m - (rng * low),
                                M + (rng * high))
        normed = norm(s.values)
        c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
        return ['background-color: %s' % color for color in c]

    with pd.option_context('display.float_format', lambda x: "%g" % x):
        df = pd.DataFrame(m)
        # s = df.style.background_gradient(cmap='Blues', high=1.4, axis=None)
        s = df.style.apply(background_gradient,
                           cmap='RdBu',
                           m=df.min().min(),
                           M=df.max().max(),
                           low=0,
                           high=0)
        display(s)


def show_dataframe(df, title=None, formats=None, headers=None, styler=None, styler_kwargs=None):
    if title is not None:
        display(HTML('<font size="4"><strong>{}</strong></font>'.format(title)))

    def check_formats(formats, headers):
        for k in formats:
            if k in headers:
                formats[headers[k]] = formats.pop(k)
        return formats

    df_ = df.copy()

    if headers is not None:
        df_.rename(columns=headers, inplace=True)

    if styler is not None:
        if headers is not None:
            if 'column' in styler_kwargs and styler_kwargs['column'] in headers:
                styler_kwargs['column'] = headers[styler_kwargs['column']]
        try:
            dfs = df_.style.apply(styler, **styler_kwargs)
        except:
            dfs = df_.style.applymap(styler, **styler_kwargs)
    else:
        dfs = df_.style

    if formats is not None:
        if headers is not None:
            formats = check_formats(formats, headers)
        dfs = dfs.format(formats)

    dfs = dfs.set_table_styles(
        [{'selector': 'td',
          'props': [('font-family', '"DejaVu Sans Mono" Consolas mono')]}])
    display(dfs)


def show_colored_matrix_test(m):
    fig, ax = plt.subplots()
    ax.matshow(m, cmap=plt.cm.Blues)

    for i in range(15):
        for j in range(15):
            c = m[j, i]
            ax.text(i, j, str(c), va='center', ha='center')

    plt.show()


def highlight_bash(data=None, file=None):
    """For use inside an IPython notebook: given a filename, print the source code. Bash version."""

    from pygments import highlight
    from pygments.lexers import BashLexer
    from pygments.formatters import HtmlFormatter
    from IPython.core.display import HTML

    if file:
        with open (file, "r") as myfile:
            data = myfile.read()

    return HTML(highlight(data, BashLexer(), HtmlFormatter(full=True)))


if __name__ == '__main__':
    size = (15, 15)

    m = np.random.randn(15 * 15).reshape(size)

    show_colored_matrix(m)
