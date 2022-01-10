import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import os
import csv

_QUASI_ZERO = 1E-6


def _inverse(m: np.ndarray):
    try:
        minv = np.linalg.inv(m)
    except np.linalg.LinAlgError as err:
        print(f'{str(err)}: Using Pseudo-inverse for least-squares solution')
        minv = np.linalg.pinv(m)
    return minv


def load_pickings(pickfile, verbose=False):
    df = pd.read_csv(pickfile,
                     delimiter=";",
                     usecols = ['event', 'station', 'channel', 'tP', 'tS', 'tPerr', 'tSerr'],
                     # Note: "tPerr" and "tSerr" are std. deviation (i.e. not variance!) of picking errors
                     skipinitialspace=True,
                     dtype={"event": str})
    for key in ['tPerr', 'tSerr']:
        if np.any(df[key].values == 0.0):
            print(f'Warning! Some {key} values are equal to zero --> replaced by {_QUASI_ZERO}')
            i0 = df[key] == 0.0
            df.loc[i0, key] = _QUASI_ZERO
    if verbose:
        df.info()
    return df


def load_delays(delayfile, verbose=False):
    df = pd.read_csv(delayfile,
                     delimiter=";",
                     usecols = ['evt1', 'evt2', 'station', 'channel', 'dtP', 'dtS', 'dtPerr', 'dtSerr'],
                     # Note: "dtPerr" and "dtSerr" are std. deviation (i.e. not variance!) of inter-event delays
                     skipinitialspace=True,
                     dtype={"evt1": str, "evt2": str, "station": str, "channel": str})
    for key in ['dtPerr', 'dtSerr']:
        if np.any(df[key].values == 0.0):
            print(f'Warning! Some {key} values are equal to zero --> replaced by {_QUASI_ZERO}')
            i0 = df[key] == 0.0
            df.loc[i0, key] = _QUASI_ZERO
    df['dtPvar'] = df['dtPerr'].values**2
    df['dtSvar'] = df['dtSerr'].values**2
    if verbose:
        df.info()
    return df


def load_data(filename, datatype='delays', verbose=False):
    """
        Load input data (pickings or delays).

        :param filename: str, Path to file
        :param filetype: str, Type of data. Can be 'pickings' (for arrival time pickings) or 'delays' (for arrival time delays)
        :param verbose, bool, verbosity flag
        :return: pandas.DataFrame objects (1 if datatype="delays", 2 if datatype="pickings")
    """
    if datatype == 'delays':
        df = load_delays(filename, verbose=verbose)
        return df
    elif datatype == 'pickings':
        p = load_pickings(filename, verbose=verbose)
        df = pickings2delays(p)
        return df, p
    else:
        raise ValueError(f'Unrecognized type of input data: "{datatype}"')


def pickings2delays(df):
    """
    Convert pickings of arrival times to inter-event delays

    :param df: Dataframe of P & S picks, as loaded using load_pickings() function
    :return: pandas.DataFrame object (same format as returned by the load_delays() function)
    """
    evtnames = pd.unique(df['event']).tolist()
    grp = df.groupby('event')
    delays = {
        'evt1': [],
        'evt2': [],
        'station': [],
        'channel': [],
        'dtP': [],
        'dtS': [],
        'dtPerr': [],
        'dtSerr': [],
        'dtPvar': [],
        'dtSvar': []
    }
    ne = len(evtnames)
    for i1 in range(ne):
        name1 = evtnames[i1]
        for i2 in range(i1 + 1, ne):
            name2 = evtnames[i2]
            indexes1 = grp.groups[name1]
            indexes2 = grp.groups[name2]

            for j1 in indexes1:
                sta1 = df.loc[i1, 'station']
                if sta1 in df.loc[indexes2, 'station'].values:
                    j2 = df.loc[indexes2, 'station'].tolist().index(sta1)
                    j2 = indexes2[i2]
                    # Check if pickings are available:
                    p_arr = [
                        np.bool(df.loc[j1, 'tP'] > 0),
                        np.bool(df.loc[j2, 'tP'] > 0),
                    ]
                    s_arr = [
                        np.bool(df.loc[j1, 'tS'] > 0),
                        np.bool(df.loc[j2, 'tS'] > 0)
                    ]
                    if np.all(p_arr) or np.all(s_arr):
                        delays['evt1'].append(name1)
                        delays['evt2'].append(name2)
                        delays['station'].append(sta1)
                        if df.loc[j1, 'channel'] == df.loc[j2, 'channel']:
                            channel = df.loc[j1, 'station']
                        else:
                            channel = f"{df.loc[j1, 'station']}-{df.loc[j2, 'station']}"
                        delays['channel'].append(channel)

                        if np.all(p_arr):
                            delays['dTP'].append(df.loc[j1, 'tP'] - df.loc[j2, 'tP'])
                            delays['dtPerr'].append(df.loc[j1, 'tPerr'] + df.loc[j2, 'tPerr'])  # std. deviation
                            delays['dtPvar'].append( (df.loc[j1, 'tPerr'] + df.loc[j2, 'tPerr'])**2 )  # variance
                        else:
                            delays['dtP'].append(_QUASI_ZERO)
                            delays['dtPerr'].append(_QUASI_ZERO)
                            delays['dtPvar'].append(_QUASI_ZERO)

                        if np.all(s_arr):
                            delays['dTS'].append(df.loc[j1, 'tS'] - df.loc[j2, 'tS'])
                            delays['dtSerr'].append(df.loc[j1, 'tSerr'] + df.loc[j2, 'tSerr'])  # std. deviation
                            delays['dtSvar'].append((df.loc[j1, 'tSerr'] + df.loc[j2, 'tSerr']) ** 2)  # variance
                        else:
                            delays['dtS'].append(_QUASI_ZERO)
                            delays['dtSerr'].append(_QUASI_ZERO)
                            delays['dtSvar'].append(_QUASI_ZERO)

    return pd.DataFrame(delays)


def iter_over_event_pairs(df, evtnames, nstamin):
    """
    Loop over all event pairs and return P & S traveltime delays
    for all pairs matching (i) event names and, (ii) minimum number
    of common stations given in input.

    :param df: Dataframe of P & S arrival time delays, as loaded using load_delays() or pickings2delays() functions
    :param evtnames: List of event names of interest
    :param nstamin: Minimum number of common stations
    :return: name1, name2, station, dtp, dts
    """
    ne = len(evtnames)
    for i1 in range(ne):
        name1 = evtnames[i1]
        for i2 in range(i1+1,ne):
            name2 = evtnames[i2]

            ns = 0
            dtp = []
            dts = []
            dtpvar = []
            dtsvar = []
            station = []
            for _, row in df[(df['evt1'] == name1) & (df['evt2'] == name2)].iterrows():
                if np.all( [np.bool(row['dtP'] > 0.0), np.bool(row['dtS'] > 0.0)] ):
                    # Increase the counter of common stations:
                    ns += 1
                    station.append(row['station'])
                    dtp.append(row['dtP'])
                    dts.append(row['dtS'])
                    dtpvar.append(row['dtPvar'])
                    dtsvar.append(row['dtSvar'])

            if ns>nstamin:
                dtp = np.array(dtp)
                dts = np.array(dts)
                dtpvar = np.array(dtpvar)
                dtsvar = np.array(dtsvar)
                yield name1, name2, station, dtp, dts, dtpvar, dtsvar


def list_available_stations(df, verbose=False):
    """
    List uniquely stations available in Dataframe
    :param df: Dataframe of P & S picks, as loaded using load_pickings() method
    :param verbose: True/False
    :return: list of unique station names
    """
    list_sta = df['station'].unique()
    sta_sorted = np.sort(list_sta).tolist()
    if verbose:
        nsta = len(sta_sorted)
        print(f'List of available stations ({nsta}):\n{sta_sorted}')
    return sta_sorted


def count_stations_per_pair(df, stations, nstamin_per_evt=2,
                            nstamin_per_event_pair=0):

    evtnames = np.unique(
        np.append(pd.unique(df['evt1']).values,
                  pd.unique(df['evt1']).values)
    )

    nevt = len(evtnames)
    print(f'{len(evtnames)} events have at least {nstamin_per_evt} records')

    # Initialize  matrix of station counts per event pair:
    num_records = np.zeros((nevt, nevt))
    evt_records = {s: np.zeros((nevt, nevt))
                   for s in stations}

    # Count station for each pair:
    for evt1, evt2, stations4pair, dtp, dts, _, _ in iter_over_event_pairs(df,
                                                                evtnames,
                                                                nstamin_per_event_pair):
        ie_1 = evtnames.index(evt1)
        ie_2 = evtnames.index(evt2)
        ns = len(stations4pair)
        num_records[ie_1, ie_2] = ns
        num_records[ie_2, ie_1] = ns
        for s in stations4pair:
            evt_records[s][ie_1, ie_2] = 1
            evt_records[s][ie_2, ie_1] = 1

    return num_records, evt_records


def get_event_names_in_pickings(df, nstamin_per_evt=0):
    """
    Return a list of event names
    :param df: Dataframe of P & S picks, as loaded using load_pickings() method
    :param nstamin_per_evt:
    :return:
    """
    evtnames = [name
                for name, subdf in df.groupby('event')
                if len(subdf) >= nstamin_per_evt]
    return evtnames


def get_event_names_in_delays(df, nstamin_per_evt=0):
    """
    Return a list of event names
    :param df: Dataframe of P & S picks, as loaded using load_pickings() method
    :param nstamin_per_evt:
    :return:
    """
    evts = []
    for _, row in df.iterrows():
        evts.append(row['evt1'])
        evts.append(row['evt2'])
    
    uniques = np.unique(evts)
    counts = np.zeros_like(uniques)
    for u in range(len(uniques)):
        counts[u] = evts.count(uniques[u])
    
    return uniques[ counts>=nstamin_per_evt ]


def get_event_names(df, datatype='pickings', nmin_sta_per_evt=0):
    if datatype == 'pickings':
        names =_get_event_names_in_pickings(df, nstamin_per_evt=nmin_sta_per_evt)
    elif datatype == 'delays':
        names = _get_event_names_in_delays(df, nstamin_per_evt=nmin_sta_per_evt)
    else:
        raise ValueError(f'Unrecognized data type: {datatype}')
    return names


def get_dates_from_pickings(df, nstamin_per_evt=0):
    """
    Return a numpy array of event dates
    :param df: Dataframe of P & S picks, as loaded using load_pickings() method
    :param nstamin_per_evt:
    :return:
    """
    evtdates = np.array([subdf.loc[subdf['tP'] > 0, 'tP'].min()
                         for name, subdf in df.groupby('event')
                         if len(subdf) >= nstamin_per_evt])
    return evtdates


def count_tt_delay_pairs(df, evtnames, nstamin_per_event_pair, min_delay):
    count = 0
    for evt1, evt2, stations, dtp, dts, _, _ in iter_over_event_pairs(df,
                                                                evtnames,
                                                                nstamin_per_event_pair):
        if np.any(np.abs(dtp-dtp.mean())>min_delay): 
            # Note added "np.abs" above on Jan 7, 2022 (Hope it fixes a bug, but not 100% sure!!)
            count += len(dtp)
    return count


def build_matrices_for_inversion(df,
                                 evtnames,
                                 stations_used,
                                 vpvsratio,
                                 nstamin_per_event_pair=2,
                                 min_delay=0.0,
                                 verbose=False,
                                 stations_wo_error=[]):
    """
    :param df: Input DataFrame
    :param evtnames: list of event (names) used in the inversion
    :param stations_used: list of stations used in the inversion
    :param vpvsratio: float, vp/vs ratio
    :param nstamin_per_event_pair: int, minimum number of stations per event pair
    :param min_delay: float, minimum accepted P-S demeaned traveltime delay
    :param verbose: boolean, set verbosity
    :param stations_wo_error: list of stations forced to have no timing errors
    :return:
    """
    d = []
    dcov = []
    vardt = []
    g = []
    d_indx = []
    """
    Note:
      d_indx: indices of each element in d
          (ievt1, ievt2, ista) for each traveltime delay
      followed by
          (i12*pol12, i23*pol23, i31*pol31) for each triplet
    """
    print(f'Counting the number of traveltime delay pairs: ', end='')
    nm = count_tt_delay_pairs(df,
                              evtnames,
                              nstamin_per_event_pair,
                              min_delay)
    print(f'{nm}')
    mcov = np.ones((nm,))/_QUASI_ZERO  # equiv. infinite a priori variance (undetermined)
    idx = 0
    # a- For each event pair at each station, add a line in d and G:
    for evt1, evt2, stations, dtp, dts, dtpvar, dtsvar in \
            iter_over_event_pairs(df, evtnames, nstamin_per_event_pair):
        dtp_dm = dtp - dtp.mean()
        dts_dm = dts - dts.mean()
        pvar = dtpvar + np.power(1/dtp.size,2)*np.sum(dtpvar)  # Variance on de-meaned P arrival time delays
        svar = dtsvar + np.power(1/dts.size,2)*np.sum(dtsvar)  # Variance on de-meaned S arrival time delays
        if verbose:
            print(f'\n# Event pair: {evt1} - {evt2}')
            print(f'Common stations with TPg and TSg pickings: {stations}')
            print(f'Diff. traveltimes:\ndTp: {dtp}\ndTs: {dts}')
            print(f'Mean diff. traveltimes:\n<dTp>: {dtp.mean()}\n<dTs>: {dts.mean()}')
            print(f'De-meaned diff. traveltimes:\ndTp-<dTp>: {dtp_dm}\ndTs-<dTs>: {dts_dm}')

        n12 = len(dtp_dm)  # Number of traveltime delays for this pair of events: (evt1, evt2)
        nex = len([s for s in stations if s in stations_wo_error])

        #if np.any(dtp_dm > min_delay):
        if n12 > 0:
            ie1 = evtnames.index(evt1)
            ie2 = evtnames.index(evt2)
            ista = [stations_used.index(s) for s in stations]
            ns = len(ista)

            for k in range(ns):
                # Add element to d (array of observations):
                d.append((dts_dm[k] - vpvsratio * dtp_dm[k]) / (1 - vpvsratio))
                dcov.append( (svar[k] + (vpvsratio**2)*pvar[k]) / (1 - vpvsratio)**2 )  # Data variance
                vardt.append(dtsvar[k])
                d_indx.append((ie1, ie2, ista[k]))

                # Buildup G:
                g_line = np.zeros((nm,))
                g_line[idx + k] = 1
                # De-meaned timing error: remove average timing error for all observations of this event pair
                g_line[idx:(idx + n12)] -= 1 / n12
                g.append(g_line)

                # b- Add constraint on stations forced to have zero timing errors (+/- picking errors):
                if stations_used[k] in stations_wo_error:
                    d.append(0.0)
                    dcov.append( (svar[k] + (vpvsratio**2)*pvar[k]) / (1 - vpvsratio)**2 )  # Data variance
                    d_indx.append((-9, -9, -9))  # Add flag to delays forced to 0
                    g_line = np.zeros((nm,))
                    g_line[idx + k] = 1
                    g.append(g_line)
                    mcov[idx + k] = vardt[k] # A priori variance on delay parameter (in array m)
       
            idx += n12
            if verbose:
                print(f'--> added {n12} S & P traveltime delay pairs to matrix d')
        elif verbose:
            print(f'--> no demeaned delays with value above {min_delay} s.')
    ndel = len(d_indx)

    # c- Add closure relationship for every earthquake triplet at each station:
    print(f'\n\nAppend station-wise closure relations for every earthquake triplet')
    d_indx = np.array(d_indx)  # Convert list of 1-D arrays to 2-D array
    i0 = np.where(d_indx[:, 2] == -9)[0]  # Find d_indx elements corresponding to forced zero delays
    d_indx_wo_zeros = np.delete(d_indx, i0, axis=0)  # Copy of d_indx and dcov without lines forced to zero delays
    dcov_indx_wo_zeros = np.delete(dcov, i0, axis=0)
    for k in range(len(stations_used)):
        j = np.where(d_indx_wo_zeros[:, 2] == k)[0]
        # Extract unique list of events belonging to pairs recorded by station k:
        evts_indices = np.unique(d_indx_wo_zeros[j, 0:2])
        all_triplets = list(itertools.combinations(evts_indices.tolist(), 3))

        # For each triplet, add a line to G, if all delays are existing in d:
        cnt_triplets = 0
        for i1, i2, i3 in all_triplets:

            def _find_evt_pair_index(e1, e2):
                ind = np.where(np.logical_and(d_indx_wo_zeros[j, 0] == e1,
                                              d_indx_wo_zeros[j, 1] == e2))[0]
                polarity = 1
                if len(ind) == 0:
                    ind = np.where(np.logical_and(d_indx_wo_zeros[j, 0] == e2,
                                                  d_indx_wo_zeros[j, 1] == e1))[0]
                    polarity = -1
                if len(ind) == 0:
                    polarity = 0
                    return None, polarity
                else:
                    return j[ind][0], polarity  # index in g_line, delay polarity (1/-1)

            i12, pol12 = _find_evt_pair_index(i1, i2)
            i23, pol23 = _find_evt_pair_index(i2, i3)
            i31, pol31 = _find_evt_pair_index(i3, i1)

            if np.all(np.array([pol12, pol23, pol31]) != 0):
                cnt_triplets += 1

                # add line to g: (delta_12 + delta_23 + delta_31 = 0):
                g_line = np.zeros((nm,))
                g_line[i12] = pol12
                g_line[i23] = pol23
                g_line[i31] = pol31
                g.append(g_line)

                # add 0 element to d, and also to the original d_indx:
                d.append(0.0)
                #dcov.append(vardt[i12] + vardt[i23] + vardt[i31])  # Variance on closure relation
                dcov.append(dcov[i12] + dcov[i23] + dcov[i31])  # Variance on closure relation
                d_indx = np.vstack([d_indx, np.array([[i12 * pol12, i23 * pol23, i31 * pol31]])])
            """
            else:
                if pol12==0:
                    print(f'    Missing traveltime delay for pair ({i1},{i2})')
                if pol23==0:
                    print(f'    Missing traveltime delay for pair ({i2},{i3})')
                if pol31==0:
                    print(f'    Missing traveltime delay for pair ({i3},{i1})')
            """
        print(f'-- station {stations_used[k]}: {cnt_triplets} triplets '+
              f'added (out of {len(all_triplets)})')

    g = np.array(g)
    d = np.array(d)
    Cd = np.diag(dcov)
    Cm = np.diag(mcov)
    print(f'Dimensions of array d: {d.shape}')
    print(f'Dimensions of matrix G: {g.shape}')
    return g, d, d_indx, Cd, Cm, ndel

"""
def solve_least_squares(G, d, Cd, Cm):
    ""
    Solves for m in the least squares problem "G.m = d", where Cd is the covariance matrix on d.
    :param G: matrix with NxM elements
    :param d: matrix with Nx1 elements: input data
    :param Cd: data covariance matrix, with NxN elements
    :return: m: solution array, with Mx1 elements
    ""
    Gt = G.transpose()
    k = G @ Cm @ Gt + Cd
    kinv = _inverse(k)
    u = Gt @ kinv
    Ginv = Cm @ u  #  Nowack & Lutter, 1988 (eqn. 1b)
    # Maximum-likelihood solution:
    m = Ginv @ d
    # Resolution matrix:
    R = Ginv @ G
    Rcomp = np.eye(R.shape[0])-R
    # Compute a posteriori covariance on parameters (Nowack & Lutter, 1988: eqn. 8)
    Cm_post = Ginv @ Cd @ Ginv.transpose() + Rcomp @ Cm @ Rcomp.transpose()
    return m, Cm_post
"""

def solve_least_squares(G, d, Cd, Cm):
    """
    Solves for m in the least squares problem "G.m = d", where Cd is the covariance matrix on d.
    :param G: matrix with NxM elements
    :param d: matrix with Nx1 elements: input data
    :param Cd: data covariance matrix, with NxN elements
    :return: m: solution array, with Mx1 elements
    """
    Gt = G.transpose()
    Cdinv = _inverse(Cd)
    Cminv = _inverse(Cm)
    k = (Gt @ Cdinv @ G + Cminv)
    kinv = _inverse(k)
    Ginv = kinv @ Gt @ Cdinv  #  Nowack & Lutter, 1988 (eqn. 1a)
    # Maximum-likelihood solution:
    m = Ginv @ d
    # Resolution matrix:
    R = Ginv @ G
    # Compute a posteriori covariance on parameters (Nowack & Lutter, 1988: eqn. 6)
    Cm_post = _inverse(Gt @ Cdinv @ G + Cminv)
    return m, Cm_post


def compute_residuals(G,d,m):
    """
    Compute residuals associated with solution m.
    Square residuals are obtained by computing (G.m - d)**2

    returns: sum_sq, rms: sum of squared residuals, root mean square
    """
    sq = np.power(G@m - d,2)
    return np.sum(sq), np.sqrt(np.mean(sq))


def m_to_station_error_matrix(m, d_indx, stations, nevt):
    tau = dict()
    # Remove elements of d_indx associated with fixed zero-delays:
    i0 = np.where(d_indx[:, 2] == -9)[0]
    d_indx_copy = np.delete(d_indx, i0, axis=0)
    for i in range(len(stations)):
        staname = stations[i]
        idat = np.where(d_indx_copy[:, 2] == i)[0]
        tau.update({staname: np.zeros((nevt, nevt))})
        for k in idat:
            # t1 - t2:
            tau[staname][d_indx_copy[k, 0], d_indx_copy[k, 1]] = m[k]
            # Opposite sign for t2-t1:
            tau[staname][d_indx_copy[k, 1], d_indx_copy[k, 0]] = -m[k]
    return tau


def pairwise_delays_to_histories(dt, d_indx, stations, nm, evtdates, Cd):
    """
    Return timing error histories from inter-event pairwise timing delays, by
    solving the least squares problem station-wise.
    :param dt: Inter-event station timing delays, in s.
    :param d_indx:
    :param stations:
    :param nm:
    :param evtdates:
    :param Cd: np.ndarray, Data covariance matrix
    :return:
    """
    histories = dict()
    # Remove elements of d_indx associated with fixed zero-delays:
    i0 = np.where(d_indx[:, 2] == -9)[0]
    d_indx_copy = np.delete(d_indx, i0, axis=0)
    assert d_indx_copy.shape[0] == Cd.shape[0]
    for i in range(len(stations)):
        idt = list()
        Gsta = []
        dsta = []
        staname = stations[i]
        # indices of all inter-event timing delays associated with the current station index:
        idat = np.where(d_indx_copy[:, 2] == i)[0]
        for k in idat:
            i1 = d_indx_copy[k, 0]  # index of event 1
            i2 = d_indx_copy[k, 1]  # index of event 2
            G_line = np.zeros((nm,))  # time-history sampled at each event occurrence time
                                        # (i.e. NM time samples)
            G_line[i1] = 1
            G_line[i2] = -1
            Gsta.append(G_line)
            dsta.append(dt[k])
            idt.append(k)

        # Initialize starting conditions:
        mcov = np.ones((nm,))/_QUASI_ZERO # Unconstrained (i.e. "infinite") a priori variance
        G_line = np.zeros((nm,))
        G_line[0] = 1
        Gsta.append(G_line)
        dsta.append(0.0)
        #mcov[0] = 0.0
        mcov[0] = _QUASI_ZERO
        # Build covariance matrix:
        dcov = np.zeros((len(idt) + 1, len(idt) + 1))
        for j1 in range(len(idt)):
            for j2 in range(j1, len(idt)):
                dcov[j1, j2] = Cd[j1, j2]
                if j1 != j2:
                    dcov[j2, j1] = Cd[j1, j2]
        dcov[-1, -1] = np.max(0.5*np.diag(Cd[:-1, :-1]))  # variance for the initial condition
        # Remove Gsta columns with only 0's:
        Gsta = np.array(Gsta)
        dsta = np.array(dsta)
        used_cols = np.any(Gsta != 0, axis=0)  # [i for i in range(Gsta.shape[1])] #
        Cm = np.diag(mcov[used_cols])
        timing_error, timing_cov = solve_least_squares(Gsta[:, used_cols], dsta, dcov, Cm)
        utcdates = np.array([np.datetime64(datetime.utcfromtimestamp(ts))
                             for ts in evtdates[used_cols]
                             ])
        histories.update({staname: {'T_UTC_in_s': evtdates[used_cols],
                                    'delay_in_s': timing_error,
                                    'std_in_s': np.sqrt(np.diag(timing_cov)),
                                    'T_UTC': utcdates}})
    return histories


# def pairwise_delays_to_histories(dt, d_indx, stations, nevt, evtdates, Cd):
#     """
#     Return timing error histories from inter-event pairwise timing delays, by
#     solving the least squares problem.
#     :param dt: Inter-event station timing delays, in s.
#     :param d_indx:
#     :param stations:
#     :param nevt:
#     :param evtdates:
#     :param Cd: np.ndarray, Data covariance matrix
#     :return:
#     """
#     histories = dict()
#     # Remove elements of d_indx associated with fixed zero-delays:
#     i0 = np.where(d_indx[:, 2] == -9)[0]
#     d_indx_copy = np.delete(d_indx, i0, axis=0)
#     assert d_indx_copy.shape[0] == Cd.shape[0]
#
#     idt = list()
#     Gsta = []
#     dsta = []
#     nsta = len(stations)
#     ista = np.array([])
#     for i in range(nsta):
#         ista = np.hstack((ista, i*np.ones(nevt,)))
#         # indices of all inter-event timing delays associated with the current station index:
#         idel = np.where(d_indx_copy[:, 2] == i)[0]
#         for k in idel:
#             i1 = d_indx_copy[k, 0]  # index of event 1
#             i2 = d_indx_copy[k, 1]  # index of event 2
#             G_line = np.zeros((nevt*nsta,))  # time-history sampled at each event occurrence time
#                                              # (i.e. NEVT time samples)
#             G_line[(i*nsta)+i1] = 1
#             G_line[(i*nsta)+i2] = -1
#             Gsta.append(G_line)
#             dsta.append(dt[k])
#             idt.append(k)
#
#     # Initialize starting conditions:
#     for i in range(nsta):
#         G_line = np.zeros((nevt*nsta,))
#         G_line[i*nsta] = 1
#         Gsta.append(G_line)
#         dsta.append(0.0)
#
#     # Build delay covariance matrix:
#     ndt = len(idt)
#     dcov = np.zeros((ndt + nsta, ndt + nsta))
#     for j1 in range(ndt):
#         for j2 in range(j1, ndt):
#             dcov[j1, j2] = Cd[j1, j2]
#             if j1 != j2:
#                 dcov[j2, j1] = Cd[j1, j2]
#     for i in range(nsta):
#         dcov[ndt+i, ndt+i] = (_QUASI_ZERO)  # approx. zero variance for the initial condition
#
#     # Remove Gsta columns with only 0's:
#     Gsta = np.array(Gsta)
#     dsta = np.array(dsta)
#     used_cols = [i for i in range(Gsta.shape[1])] #np.any(Gsta != 0, axis=0)
#     ista = ista[used_cols]
#     tiledates = np.tile(evtdates, nsta)[used_cols]
#     utcdates = np.array([np.datetime64(datetime.utcfromtimestamp(ts))
#                          for ts in tiledates
#                          ])
#     timing_error, timing_cov = solve_least_squares(Gsta[:, used_cols], dsta, dcov)
#     timing_std = np.sqrt(np.diag(timing_cov))
#     for i in range(nsta):
#         staname = stations[i]
#         j = np.where(ista == i)[0]
#         histories.update({staname: {'T_UTC_in_s': tiledates[j],
#                                     'delay_in_s': timing_error[j],
#                                     'std_in_s': timing_std[j],
#                                     'T_UTC': utcdates[j]}})
#     print(histories)
#     return histories


def build_demeaned_delays(df,
                          evtnames,
                          nstamin_per_event_pair,
                          verbose=False,
                          max_abs_delay=None):
    """
    Compute de-meaned inter-event traveltime delays for all pairs of events
    """
    if max_abs_delay is None:
        max_abs_delay = np.inf
    dtp = []
    dts = []
    for evt1, evt2, stations, dtp_pair, dts_pair, _, _ in \
            iter_over_event_pairs(df, evtnames, nstamin_per_event_pair):
        dtp_dm = dtp_pair - dtp_pair.mean()
        dts_dm = dts_pair - dts_pair.mean()
        if verbose:
            print(f'\n# Event pair: {evt1} - {evt2}')
            print(f'Common stations with TPg and TSg pickings: {stations}')
            print(f'Diff. traveltimes:\ndTp: {dtp_pair}\ndTs: {dts_pair}')
            print(f'Mean diff. traveltimes:\n<dTp>: {dtp_pair.mean()}\n<dTs>: {dts_pair.mean()}')
            print(f'De-meaned diff. traveltimes:\n[dTp]: {dtp_dm}\n[dTs]: {dts_dm}')

        # Append demeaned diff. traveltimes to the global array:
        if np.abs(dts_dm).max() < max_abs_delay:
            dtp += list(dtp_dm)
            dts += list(dts_dm)
    if verbose:
        print(f'\nTotal number of (dTP,dTs) observations: ({len(dtp)},{len(dts)})')
    return dtp, dts


def write_timing_errors(outputdir, stations, tehdict):
    for sta in stations:
        filename = os.path.join(outputdir, f'timing_delays_{sta}.txt')
        with open(filename, 'wt', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(('T_UTC', 'T_UTC_in_s', 'delay_in_s', 'std_in_s'))
            writer.writerows(zip(tehdict[sta]['T_UTC'],
                                 tehdict[sta]['T_UTC_in_s'],
                                 tehdict[sta]['delay_in_s'],
                                 tehdict[sta]['std_in_s']))


def write_residuals(outputdir, rms, sum_sq_res):
    filename = os.path.join(outputdir, 'residuals.txt')
    with open(filename, 'wt') as f:
         f.write(f'RMS = {rms}\n')
         f.write(f'SUM SQ. RES. = {sum_sq_res}')

