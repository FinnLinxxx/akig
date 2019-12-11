from os.path import normpath, join

from numpy import cos, sin, arctan, arctan2, arcsin, array, pi, sqrt, random, isscalar, vstack, nanmin, nanmax, mat, \
    matrix, multiply, ndarray, hstack
import pandas as pd

from utils.rosutils import load_rosbag_csv, convert_timestamp

import matplotlib.pyplot as plt

RHO = pi / 200


def euler2dcm(eul, order='xyz'):
    """
    Compute C_b^n from Psi_nb
    :param eul: Psi_nb (from n to b)
    :return: C_b^n (from b to n)
    """
    cr = cos(eul[0, 0])
    sr = sin(eul[0, 0])
    cp = cos(eul[1, 0])
    sp = sin(eul[1, 0])
    cy = cos(eul[2, 0])
    sy = sin(eul[2, 0])

    if order == 'xyz':
        dcm = array([[cp * cy, (sp * sr * cy - cr * sy), (cr * sp * cy + sr * sy)],
                     [cp * sy, (sr * sp * sy + cr * cy), (cr * sp * sy - sr * cy)],
                     [-sp, sr * cp, cr * cp]])
    elif order == 'zyx':
        dcm = array([[cp * cy, -sy * cp, sp],
                    [sp * sr * cy + sy * cr, -sp * sr * sy + cr * cy, -sr * cp],
                    [-sp * cr * cy + sr * sy, sp * sy * cr + sr * cy, cp * cr]])
    return dcm


def transform_eul(eul, rot):
    Cab = euler2dcm(rot)
    return dcm2euler(Cab @ euler2dcm(eul))


def angle(x):
    if x > pi:
        x -= 2 * pi
    if x < -pi:
        x += 2 * pi
    return x


def angles(x):
    x[x > pi] -= 2*pi
    x[x < -pi] += 2*pi
    return x


def dcm2euler(dcm, skip_check=False):
    """
    Compute Psi_nb from C_b^n
    :param dcm: C_b^n (from b to n)
    :return: Psi_nb (from n to b)
    """
    roll = arctan2(dcm[2, 1], dcm[2, 2])
    pitch = arcsin(-dcm[2, 0])
    yaw = arctan2(dcm[1, 0], dcm[0, 0])

    singular_tolerance = 1e-6
    if pitch < 0:
        singular_check = abs(pitch + pi / 2)
    else:
        singular_check = abs(pitch - pi / 2)

    if not skip_check and singular_check < singular_tolerance:
        raise ValueError('Euler angle singularity at pitch=90 degrees encountered.')

    return array([[roll], [pitch], [yaw]])


def sim_random_constant(z_true, var, count):
    """ Simulate a random contant with specified variance for a number of epochs

    :param z_true: the true value
    :type z_true: float
    :param count: number of epochs to be simulated
    :type count: int
    :param var: variance of the observation values
    :type var: float
    :return: Drawn samples from the parametrized normal distribution
    :rtype: :py:class:`numpy.ndarray` of shape (n,)
    """
    # TODO: support multiple random constants
    z = random.normal(z_true, sqrt(var), size=(count,))
    return z


def sim_random_vector(z_true, var, count):
    if isscalar(var):
        fs = random.normal(z_true, sqrt(var), size=(z_true.shape[0], count))
    else:
        fs = random.normal(z_true, sqrt(var), size=(z_true.shape[0], count))
    # TODO: support full VCM (correlated simulation

    return fs


def iqr_outlier(df, column, th=1.5, width=100):
    q1 = df[column].rolling(width, center=True).quantile(0.25)
    q3 = df[column].rolling(width, center=True).quantile(0.75)
    iqr = q3 - q1

    # print(nanmin(df[column].values)+nanmax(iqr.values)*th)
    q3.fillna(nanmin(df[column].values)-nanmax(iqr.values)*th, inplace=True)
    iqr.fillna(0., inplace=True)
    # print((q3 + th * iqr))

    outlier = (df[column] < (q1 - th * iqr)) | (df[column] > (q3 + th * iqr))
    return outlier


def select_quantile(df, column, q, upper=False, width=100):
    q = df[column].rolling(width, center=True).quantile(q)
    if upper:
        q.fillna(df[column].max(), inplace=True)
        inlier = df[column] > q
    else:
        q.fillna(df[column].min(), inplace=True)
        inlier = df[column] < q
    return inlier


def format_axis(ax):
    # bugfix for sans-serif tick labels (https://stackoverflow.com/a/51409039)
    ax.set_axisbelow(True)
    ax.xaxis.get_major_formatter()._usetex = False
    ax.yaxis.get_major_formatter()._usetex = False
    ax.grid(linestyle=':')
    ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)


def format_figure(fig=None, share_x=False, share_y=False):

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

    if fig is None:
        fig = plt.gcf()

    if share_y:
        for i, x in enumerate(list(fig.get_axes())):
            if i != 0:
                x.tick_params(labelleft=False)

    for x, has_more in lookahead(list(fig.get_axes())):
        if has_more and share_x:
            x.tick_params(labelbottom=False)
        format_axis(x)
    # plt.tight_layout()


def sphere2cart(measures, start=array([0, 0, 0])):
    """
    Convert an array of nx3 (hz, v, ss) to point coordinates (east, north, height) [nx3]
    :param measures:
    :param start:
    :return:
    """
    if type(measures) is ndarray:
        measures = mat(measures)
    elif type(measures) is not matrix:
        raise ValueError("Unsupported type of parameter 'measures'. Must be numpy.matrix or numpy.array")

    north = multiply(multiply(measures[:, 2].T, sin(measures[:, 1].T)), cos(measures[:, 0].T))
    east = multiply(multiply(measures[:, 2].T, sin(measures[:, 1].T)), sin(measures[:, 0].T))
    height = multiply(measures[:, 2].T, cos(measures[:, 1].T))

    east += start[0]
    north += start[1]
    height += start[2]

    return hstack((east.T, north.T, height.T))


def window_grouper(df_original, win_f, margin=None, offset=None):
    df_t = load_rosbag_csv(win_f)
    df_t = convert_timestamp(df_t, ['time', 'field.header.stamp', 'field.source_start', 'field.record_start',
                                    'field.source_end', 'field.record_end'],
                             ['time2', 'header_stamp', 'source_start', 'record_start', 'source_end', 'record_end'])

    df = df_original.copy()
    # df.loc[:, 'group'] = "Unknown"

    for i, r in df_t.iterrows():
        from_time = r['source_start']
        to_time = r['source_end']
        pt = r['field.name']

        if offset is not None:
            from_time += pd.Timedelta(seconds=offset)
            to_time += pd.Timedelta(seconds=offset)

        if margin:
            from_time += pd.Timedelta(seconds=margin)
            to_time -= pd.Timedelta(seconds=margin)

        df.loc[from_time:to_time, 'group'] = pt

    return df


def window_slicer(df, win_f, margin=None, offset=None, plot_columns=None, robust=None):
    if type(win_f) is pd.DataFrame:
        df_t = win_f
    else:
        df_t = load_rosbag_csv(win_f)
        df_t = convert_timestamp(df_t, ['time', 'field.header.stamp', 'field.source_start', 'field.record_start',
                                        'field.source_end', 'field.record_end'],
                                 ['time2', 'header_stamp', 'source_start', 'record_start', 'source_end', 'record_end']) 

    if plot_columns is not None:
        plt.figure()
        plt.plot(df[plot_columns], '+')

    pts = {}
    # iterate over windows
    for i, r in df_t.iterrows():
        from_time = r['source_start']
        to_time = r['source_end']
        pt = r['field.name']

        if offset is not None:
            from_time += pd.Timedelta(seconds=offset)
            to_time += pd.Timedelta(seconds=offset)

        if margin:
            from_time += pd.Timedelta(seconds=margin)
            to_time -= pd.Timedelta(seconds=margin)

        pt_df = df[from_time:to_time]

        if robust is not None:
            idx = None
            for c in robust:
                c_idx = iqr_outlier(pt_df, c)
                if idx is None:
                    idx = ~c_idx
                else:
                    idx = idx & ~c_idx
            if plot_columns is not None:
                plt.plot(pt_df[plot_columns], '+', color='C7')
                
            pt_df = pt_df[idx.values]

        if plot_columns is not None:
            plt.plot(pt_df[plot_columns], '+', color='C2')

        if not pt_df.empty:
            pts[pt] = pt_df.mean().tolist() + pt_df.var().tolist()
        else:
            print('... No data for {}'.format(pt))

    df_pts = pd.DataFrame.from_dict(pts, orient='index', columns=df.mean().keys().tolist() +
                                    [i+'_var' for i in df.mean().keys().to_list()])

    if plot_columns is not None:
        plt.plot(pd.to_datetime(df_pts['ut'], unit='s'), df_pts[plot_columns], 's', color='C3')
        format_figure()
        plt.show()

    return df_pts
