import numpy as np
import pandas as pd


def read_geosi(file):
    """
    Einleseroutine der Messdaten aus Geosi.

    - Datei ist nach Standpunkten sortiert
    - Zeilenweise die Datei durchgehen
    - eigene Funktion read_geosi()

    - Einlesen der Datein passiert Instrumentenweise (in AGPE nicht notwendig,
        aber als Vorbereitung auf GZIG)

    - Satzmittel ist bereits in Geosi gerechnet
        Berücksichtigung mehrerer Aufstellungen pro Standpunkt sollte berücksichtigt werden
        (in Feldübung implementieren) > Spalte aufst = nummer der Aufstellung

    :return: Messdaten Array (von, nach, aufst, ri, sh, dh)
    """

    def prealloc_matrix(shape, columns=None, fill_with_nan=False):
        if columns is None:
            m = np.empty(shape, dtype=np.float64)
        else:
            m = np.empty(shape, dtype=columns)
        if fill_with_nan:
            m.fill(np.NaN)
        return m

    # preallocate data matrix
    drl = 100000
    data = prealloc_matrix((drl,), columns=[('von', 'U10'), ('nach', 'U10'), ('aufst', np.int16),
                                            ('ri', np.float64), ('sh', np.float64), ('dh', np.float64)])

    # working vars
    standpunkt = None
    aufstellung = 1
    i = 0

    with open(file, 'r') as fh:
        for line in fh:
            if line.startswith('Stand'):
                # new Standpunkt
                standpunkt = line[7:].strip()
            else:
                try:
                    entries = [np.nan if e == '---' else e for e in line.split()]
                    data[i] = tuple([standpunkt, entries[0].strip(), aufstellung,
                                     np.float64(entries[1]) / 200. * np.pi, np.float64(entries[2]),
                                     np.float64(entries[3])])
                    i += 1
                except:
                    pass

    return pd.DataFrame(data[:i])


def read_punkte(file):
    pt_df = pd.read_csv(file, sep=r'\t', na_values=['---'],
                        header=None, names=['pt', 'y', 'x', 'h', 'typ'])
    pt_df.set_index('pt', inplace=True, drop=False)
    return pt_df


def get_coords(df, name1, name2=None, params=None):
    if params is not None:
        new_points = list(set(params[params.type == 'coord'].point.to_list()))
    else:
        new_points = []

    if name2 is None:
        if name1 in new_points:
            return float(params[params.name == 'y_' + name1].value), float(params[params.name == 'x_' + name1].value)
        else:
            return df.loc[name1].y, df.loc[name1].x
    else:
        if name1 in new_points:
            c_1 = (float(params[params.name == 'y_' + name1].value), float(params[params.name == 'x_' + name1].value))
        else:
            c_1 =  (df.loc[name1].y, df.loc[name1].x)
        if name2 in new_points:
            c_2 = (float(params[params.name == 'y_' + name2].value), float(params[params.name == 'x_' + name2].value))
        else:
            c_2 = (df.loc[name2].y, df.loc[name2].x)

        return c_1 + c_2


def create_params_obs(pt_df, df):
    def check_coordinates(obs_df, pt_df):
        upt = pt_df.pt.unique()
        obs = set(obs_df.von.unique()) | set(obs_df.nach.unique())

        for o in obs:
            if o not in upt:
                raise ValueError(f"unknown coordinates for point '{o}'.")

        return sorted(list(pt_df.pt[pt_df.typ == 'N'].unique())), sorted(list(pt_df.pt[pt_df.typ == 'F'].unique()))

    new_points, fix_points = check_coordinates(df, pt_df)

    unknown_os = ['o_{}'.format(i) for i in list(df.von.unique())]
    unknown_cs = [None] * len(new_points) * 2
    ys, xs = ['y_{}'.format(i) for i in new_points], ['x_{}'.format(i) for i in new_points]
    unknown_cs[::2] = ys
    unknown_cs[1::2] = xs
    p_names = unknown_cs + unknown_os
    p_types = ['coord'] * len(unknown_cs) + ['ori'] * len(unknown_os)
    p_values = [list(get_coords(pt_df, n)) for n in new_points]
    flatten = lambda l: [item for sublist in l for item in sublist]
    p_values = flatten(p_values) + [0.] * len(unknown_os)

    n = len(df['ri'][df['ri'].notna()]) + len(df['sh'][df['sh'].notna()])
    l_names, l = [None] * n, [0.] * n
    l_types, l_from, l_to = [None] * n, [None] * n, [None] * n

    j = 0

    for i, r in df.iterrows():
        if not np.isnan(r['ri']):
            # add row for direction
            l[j] = r['ri']
            l_names[j] = f"ri_{r['von']}_{r['nach']}"
            l_types[j] = 'ri'
            l_from[j], l_to[j] = r['von'], r['nach']
            j += 1

        if not np.isnan(r['sh']):
            # add row for distance
            l[j] = r['sh']
            l_names[j] = f"sh_{r['von']}_{r['nach']}"
            l_types[j] = 'sh'
            l_from[j], l_to[j] = r['von'], r['nach']
            j += 1

    p_df = pd.DataFrame({'name': p_names, 'value': p_values, 'type': p_types, 'dx': [0.] * len(p_names),
                         'point': [x.replace('x_', '').replace('y_', '').replace('o_', '') for x in p_names]})
    l_df = pd.DataFrame({'name': l_names, 'value': l, 'type': l_types, 'v': [0.] * len(l_names),
                         'von': l_from, 'nach': l_to})

    return p_df, l_df


def netz2d(pt_df, l_df, x_df):
    """
    Eigenständige Funktion zum Punktmanagement im Netz -> Netzmanagement

    - haben alle in den Messdaten vorkommenden Punkte Koordinaten (fest oder näherung)
    - Identifikation der Neupunkte. An der sortierten Liste dieser Punkte orientieren sich die
        Spalten der A-Matrix (y, x)
    - Für alle Beobachtungen rufe Beobachtungsmodell auf -> Koeffizienten der A-Matrix und
        gekürzter Beobachtungsvektor
    :param pt_df:
    :param df:
    :return: A-Matrix und gekürzter Beobachtungsvektor l
    """

    def distance_obs(y1, x1, y2, x2, l):
        l_comp = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
        a1 = (y1 - y2) / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        a2 = (x1 - x2) / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        a3 = (-y1 + y2) / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        a4 = (-x1 + x2) / np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        l_short = l - l_comp

        return a1, a2, a3, a4, l_short

    def direction_obs(y1, x1, y2, x2, o1, l):
        l_comp = np.arctan2((y2 - y1), (x2 - x1)) - o1
        a1 = (x1 - x2) / ((x1 - x2) ** 2 + (y1 - y2) ** 2)
        a2 = (-y1 + y2) / ((x1 - x2) ** 2 + (y1 - y2) ** 2)
        a3 = (-x1 + x2) / ((x1 - x2) ** 2 + (y1 - y2) ** 2)
        a4 = (y1 - y2) / ((x1 - x2) ** 2 + (y1 - y2) ** 2)
        a5 = -1

        l_short = l - l_comp

        if l_short > 2 * np.pi:
            l_short -= 2 * np.pi
        if l_short < 0:
            l_short += 2 * np.pi

        if abs(l_short) > abs(l_short - 2 * np.pi):
            l_short -= 2 * np.pi

        return a1, a2, a3, a4, a5, l_short

    A = np.zeros((df_l.shape[0], df_x.shape[0]), dtype=np.float64)
    l = np.zeros((df_l.shape[0], 1), dtype=np.float64)

    j = 0
    unknowns = x_df.name.to_list()
    new_points = list(set(x_df[x_df.type == 'coord'].point.to_list()))

    for i, r in l_df.iterrows():
        if r['type'] == 'ri':
            # add row for direction
            pts = get_coords(pt_df, r['von'], r['nach'], params=x_df)
            o = float(x_df[x_df['type'] == 'ori'].value[x_df.point == r['von']])
            a0, a1, a2, a3, a4, li = direction_obs(*(pts + (o, r['value'])))

            if r['von'] in new_points:
                A[j, unknowns.index('y_' + r['von'])] = a0
                A[j, unknowns.index('x_' + r['von'])] = a1

            if r['nach'] in new_points:
                A[j, unknowns.index('y_' + r['nach'])] = a2
                A[j, unknowns.index('x_' + r['nach'])] = a3

            A[j, unknowns.index('o_' + r['von'])] = a4

            l[j, 0] = li
            j += 1

        if r['type'] == 'sh':
            # add row for distance
            pts = get_coords(pt_df, r['von'], r['nach'])
            a0, a1, a2, a3, li = distance_obs(*(pts + (r['value'],)))

            if r['von'] in new_points:
                A[j, unknowns.index('y_' + r['von'])] = a0
                A[j, unknowns.index('x_' + r['von'])] = a1
            if r['nach'] in new_points:
                A[j, unknowns.index('y_' + r['nach'])] = a2
                A[j, unknowns.index('x_' + r['nach'])] = a3

            l[j, 0] = li
            j += 1

    return A, l


if __name__ == '__main__':
    epsilon = 1e-9

    df = read_geosi(r'data/netz/obs.txt')
    pt_df = read_punkte(r'data/netz/punkte.txt')

    df_x, df_l = create_params_obs(pt_df, df)

    while True:
        A, l = netz2d(pt_df, df_l, df_x)
        d = np.zeros((len(l),), dtype=np.float64)
        d[df_l.type=='ri'] = (0.3e-3/200*np.pi)**2
        d[df_l.type=='sh'] = 0.002**2
        P = np.diag(d) / np.mean(d)

        N = A.T @ P @ A

        # parameters
        x = np.linalg.solve(N, A.T @ P @ l)
        # residuals
        v = A @ x - l

        df_x['dx'] = x
        df_l['v'] = v

        # update
        df_x['value'] = df_x['value'] + df_x['dx']

        if np.max(np.abs(x)) < epsilon:
            break

    print('OK')
