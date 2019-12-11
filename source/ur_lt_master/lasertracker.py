import matplotlib.pyplot as plt

from utils.misc import format_figure, sphere2cart
from utils.misc import euler2dcm, dcm2euler
from utils.rosutils import load_rosbag_csv
from numpy import pi, hstack, int64
import pandas as pd


def load_sa_output(txt_path, target_path=None, filtermask=None, angle_unit='rad'):
    df = pd.read_csv(txt_path, comment='#', header=None,
                     names=['name', 'x', 'y', 'z', 'ro', 'pi', 'ya'])

    # filter
    if filtermask:
        df = df[df['name'].str.contains(filtermask)]

    if angle_unit == 'gon':
        df[['ro', 'pi', 'ya']] = df[['ro', 'pi', 'ya']] * pi / 200

    # strip name
    df['name'] = df['name'].apply(lambda x: 'L' + x.rsplit('_', 1)[-1])

    # compute DCMs
    dcms = df.apply(lambda x: euler2dcm(x.iloc[4:].values[:, None], order='zyx'), axis=1)
    # convert back
    df['ro'], df['pi'], df['ya'] = dcms.apply(lambda x: dcm2euler(x)[0, 0]), \
                                   dcms.apply(lambda x: dcm2euler(x)[1, 0]), \
                                   dcms.apply(lambda x: dcm2euler(x)[2, 0])

    # sort
    df.sort_values(by='name', inplace=True)
    df.set_index('name', inplace=True)
    return df


def load_and_save(txt_path, target_path=None, plot_timestamps=False):
    df = load_rosbag_csv(txt_path)

    if plot_timestamps:
        # timestamp analysis [ms]
        t_pi = (df['field.header.stamp'] - df['field.header.stamp'][0])/1e9
        dt = (t_pi - df['field.t']) * 1e3
        plt.plot(dt, '+')
        format_figure()
        plt.show()

    t1 = pd.to_datetime(df['field.header.stamp'], unit='ns')
    t2 = pd.to_datetime(df['field.t'] + df['field.header.stamp'][0]/1e9, unit='s')

    coord = df[['field.header.stamp', 'field.t', 'field.x', 'field.y', 'field.z']].values
    ori = df[['field.rx', 'field.ry', 'field.rz']].values/200*pi

    df = pd.DataFrame(data=hstack((coord, ori)), index=[i for i in range(df.shape[0])],
                      columns=['ut', 't', 'x', 'y', 'z'] + ['ro', 'pi', 'ya'])

    df['ut'] = t2.astype(int64) / 1e9
    df['t_pi'] = t1
    df['t_lt'] = t2
    df['k'] = [i for i in range(df.shape[0])]
    df.set_index('t_lt', drop=False, inplace=True)

    if target_path is not None:
        df.to_pickle(target_path)

    return df


def load_ros_log(txt_path, version=1):
    if version == 2:
        df = pd.read_csv(txt_path, sep=' ', header=None, names=['t', 'sec', 'mus', 'hz', 'v', 'd'])
    elif version == 3:
        df = pd.read_csv(txt_path, sep=' ', header=None, names=['t', 'sec', 'mus', 'x', 'y', 'z'])
    else:
        df = pd.read_csv(txt_path, sep=' ', header=None, names=['sec', 'mus', 'hz', 'v', 'd'])

    df.dropna(inplace=True)
    df["t"] = pd.to_numeric(df["t"])
    df["sec"] = pd.to_numeric(df["sec"])
    df["mus"] = pd.to_numeric(df["mus"])
    if version == 3:
        df["x"] = pd.to_numeric(df["x"])
        df["y"] = pd.to_numeric(df["y"])
        df["z"] = pd.to_numeric(df["z"])
        df['time'] = pd.to_timedelta(df['sec'] + df['mus'] / 1e6, unit='s')
        return df
        
    df["hz"] = pd.to_numeric(df["hz"])
    df["v"] = pd.to_numeric(df["v"])
    df["d"] = pd.to_numeric(df["d"])

    df['time'] = pd.to_timedelta(df['sec'] + df['mus'] / 1e6, unit='s')
    df[['y', 'x', 'z']] = pd.DataFrame(sphere2cart(df[['hz', 'v', 'd']].values), index=df.index)
    df['y'] *= -1
    
    return df


def transform_to_robot(df, params):
    """
    Transform set of lasertracker observations to robot poses
    :param df: dataframe of lasertracker observations (must contain 'x', 'y', 'z', and dcm elements)
    :param params:
    :return:
    """
    rot_m = df[['dcm_{}'.format(i) for i in range(9)]].apply(lambda x: x.values.reshape((3,3)), axis=1)

    def convert_mv(R, mv):
        return R[0] @ mv

    mv = pd.DataFrame(rot_m).apply(lambda row: convert_mv(row, params.mv), axis=1)
    df[['mvx', 'mvy', 'mvz']] = pd.DataFrame(mv.apply(lambda x: x.flatten().tolist()).tolist(), index=mv.index)

    # add mounting
    df[['x2', 'y2', 'z2']] = pd.DataFrame(df[['x', 'y', 'z']].values + df[['mvx', 'mvy', 'mvz']].values, index=mv.index)

    def transform(lt, R_lt_ur, t_lt_ur):
        return R_lt_ur @ lt.values[:, None] + t_lt_ur

    # transform
    R = euler2dcm(params.rot_ur_lt)
    df[['xur', 'yur', 'zur']] = pd.DataFrame(df[['x2', 'y2', 'z2']].apply(
        lambda row: transform(row, R, params.t_lt_ur).flatten().tolist(), axis=1).tolist(), index=df.index)

    return df
