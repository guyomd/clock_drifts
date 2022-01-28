import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import os
import csv
import sys
from math import ceil


from clock_drifts.data import DataManager, _QUASI_ZERO



class ClockDriftEstimator(object):
    def __init__(self, dm: DataManager):
        self.dm = dm  # instance of DataManager object
        self.G = None
        self.d = None
        self.d_index = None
        self.ndel = None
        self.Cd = None
        self.Cm = None
        self.m = None
        self.sqres = None
        self.rms = None

    def _build_inputs_for_inversion(self, vpvsratio, reference_stations,
                                      add_closure_triplets=True):

        if self.dm.stations is None:
            print('!! Missing list of stations. Loading stations...')
            self.dm.list_stations()
        if self.dm.evtnames is None:
            print('!! Missing event names. Loading events...')
            self.dm.load_events()
        if self.dm.delays is None:
            print('!! Missing delays (!). Loading data...')
            self.dm.load_data()

        self.G, self.d, self.d_indx, self.Cd, self.Cm, self.ndel = \
                _build_matrices(self.dm.delays,
                                self.dm.evtnames,
                                self.dm.stations,
                                vpvsratio,
                                min_sta_per_pair=self.dm.min_sta_per_pair,
                                stations_wo_error=reference_stations,
                                add_closure_triplets=add_closure_triplets)


    def _solve_least_squares(self):
        if (self.G is not None) \
            and (self.d is not None) \
            and (self.Cd is not None) \
            and (self.Cm is not None):
            self.m, self.Cm = _solve_least_squares(
                self.G,
                self.d,
                self.Cd,
                self.Cm)
        else:
            raise ValueError('Missing at least one of the following quantities: G, d, Cd, Cm.')

    def _compute_residuals(self):
        if (self.G is not None) \
            and (self.d is not None) \
            and (self.m is not None):
            self.sqres, self.rms = _compute_residuals(
                self.G,
                self.d,
                self.m)
        else:
            raise ValueError('Missing at least one of the following quantities: G, d, m')

    def _pairwise_delays_to_histories(self):
        if (self.m is not None) \
            and (self.d_indx is not None) \
            and (self.ndel is not None) \
            and (self.Cm is not None):
            self.drifts = _pairwise_delays_to_histories(
                self.m,
                self.d_indx[:self.ndel, :],
                self.dm.stations,
                len(self.dm.evtnames),
                self.dm.evtdates,
                self.Cm)
        else:
            raise ValueError('Missing at least one of the following quantities: m, d_indx, ndel, Cm')

    def write_outputs(self, outdir):
        """
        Write inversion results in a file

        :param outdir: str, Path to output directory
        """
        _write_timing_errors(outdir, self.dm.stations, self.drifts)
        _write_residuals(outdir, self.rms, self.sqres)

    def run(self, vpvsratio, reference_stations, add_closure_triplets=True):
        print(f'\n>> [1/4] Build matrices for inversion')
        if add_closure_triplets:
            print(f'         (including closure triplets)')
        else:
            print(f'         (closure triplets not included)')
        self._build_inputs_for_inversion(
            vpvsratio,
            reference_stations,
            add_closure_triplets=add_closure_triplets)
        print(f'\n>> [2/4] Run inversion of relative drifts')
        self._solve_least_squares()
        print(f'\n>> [3/4] Compute residuals')
        self._compute_residuals()
        print(f'         sum of square residuals: {self.sqres}')
        print(f'         root mean square: {self.rms}')
        print(f'\n>> [4/4] Convert relative delays to clock drift histories')
        self._pairwise_delays_to_histories()
        return self.drifts  # Return clock drift histories as Python dict


class ProgressBar():
    """
    Progress-bar object definition
    """
    def __init__(self, imax: float, title: str='', nsym: int=20):
        """
        :param imax: float, Maximum counter value, corresponding to 100% advancment
        :param title: str, (Optional) Title string for the progress bar
        :param nsym: int, (Optional) Width of progress bar, in number of "=" symbols (default: 20)
        """
        self.imax = imax
        self.title = title
        self.nsym = nsym


    def update(self, i: float, imax: float=None, title: str=None):
        """ Display an ASCII progress bar with advancement level at (i/imax) %

        :param i: float, Current counter value
        :param imax: float, Maximum counter value, corresponding to 100% advancment
        :param title: str, (Optional) Title string for the progress bar
        :param nsym: int, (Optional) Width of progress bar, in number of "=" symbols (default: 20)
        """
        if imax is not None:
            self.imax = imax
        if title is not None:
            self.title = title
        sys.stdout.write('\r')
        fv = float(i)/float(self.imax)  # Fractional value, between 0 and 1
        sys.stdout.write( ('{0} [{1:'+str(self.nsym)+'s}] {2:3d}%').format(self.title, '='*ceil(
            fv*self.nsym), ceil(fv*100)) )
        if i==self.imax:
            sys.stdout.write('\n')
        sys.stdout.flush()



# Functions:
def _inverse(m: np.ndarray):
    try:
        minv = np.linalg.inv(m)
    except np.linalg.LinAlgError as err:
        print(f'{str(err)}: Using Pseudo-inverse for least-squares solution')
        minv = np.linalg.pinv(m)
    return minv

def _build_matrices(delays, evtnames,stations_used, vpvsratio, min_sta_per_pair=2,
                    verbose=False, stations_wo_error=[], add_closure_triplets=True):
    """
    :param delays: Pandas.DataFrame, as formatted by load_data() method
    :param evtnames: list of event (names) used in the inversion
    :param stations_used: list of stations used in the inversion
    :param vpvsratio: float, vp/vs ratio
    :param min_sta_per_pair: int, minimum number of stations per event pair
    :param verbose: boolean, set verbosity
    :param stations_wo_error: list of stations forced to have no timing errors
    :param add_closure_triplets: boolean, Flag specifying whether closure triplets should be appended to G.
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
    pairs = delays.groupby(['evt1', 'evt2']) \
                  .filter(lambda x: len(x) > min_sta_per_pair)  # Pandas.DataFrame instance
    nm = len(pairs) 
    print(f'Event pairs (arrival-time delays) recorded by at least {min_sta_per_pair} stations: {nm}')
    bar = ProgressBar(nm)
    mcov = np.ones((nm,))/_QUASI_ZERO  # equiv. infinite a priori variance (undetermined)
    idx = 0
    # For each event pair at each station, add a line in d and G:
    for grp_name, grp in pairs.groupby(['evt1', 'evt2']):
        evt1, evt2 = grp_name
        dtp_dm = (grp['dtP']-grp['dtP'].mean()).values
        dts_dm = (grp['dtS']-grp['dtS'].mean()).values
        dtpvar = grp['dtPvar'].values
        dtsvar = grp['dtPvar'].values
        n12 = len(grp['dtP'])  # Number of delays for the current pair: (evt1, evt2)
        pvar = (grp['dtPvar'] + grp['dtPvar'].sum()*np.power(1 / n12, 2)).values  # Variance on de-meaned P arrival time delays
        svar = (grp['dtSvar'] + grp['dtSvar'].sum()*np.power(1 / n12, 2)).values  # Variance on de-meaned S arrival time delays
        nex = len([s for s in grp['station'] if s in stations_wo_error])

        ie1 = evtnames.index(evt1)
        ie2 = evtnames.index(evt2)
        ista = [stations_used.index(s) for s in grp['station']]
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

            # Eventually, add constraint on stations forced to have zero timing errors (+/- picking errors):
            if stations_used[k] in stations_wo_error:
                d.append(0.0)
                dcov.append( (svar[k] + (vpvsratio**2)*pvar[k]) / (1 - vpvsratio)**2 )  # Data variance
                d_indx.append((-9, -9, -9))  # Add flag to delays forced to 0
                g_line = np.zeros((nm,))
                g_line[idx + k] = 1
                g.append(g_line)
                mcov[idx + k] = vardt[k] # A priori variance on delay parameter (in array m)
          
        idx += n12
        bar.update(idx, title='Adding delays to matrices G and d... ')
    ndel = len(d_indx)
    d_indx = np.array(d_indx)  # Convert list of 1-D arrays to 2-D array

    # Add closure relationship for every earthquake triplet at each station:
    if add_closure_triplets:
        i0 = np.where(d_indx[:, 2] == -9)[0]  # Find d_indx elements corresponding to forced zero delays
        d_indx_wo_zeros = np.delete(d_indx, i0, axis=0)  # Copy of d_indx and dcov without lines forced to zero delays
        dcov_indx_wo_zeros = np.delete(dcov, i0, axis=0)
        nsta = len(stations_used)
        bar = ProgressBar(nsta, title=f'Adding closure relations... ')
        for k in range(nsta):
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

                if np.any(np.array([pol12, pol23, pol31]) == 0):
                    continue

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

            bar.update(k+1)

    g = np.array(g)
    d = np.array(d)
    Cd = np.diag(dcov)
    Cm = np.diag(mcov)
    print(f'Dimensions of array d: {d.shape}')
    print(f'Dimensions of matrix G: {g.shape}')
    return g, d, d_indx, Cd, Cm, ndel

"""
def _solve_least_squares(G, d, Cd, Cm):
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

def _solve_least_squares(G, d, Cd, Cm):
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


def _compute_residuals(G,d,m):
    """
    Compute residuals associated with solution m.
    Square residuals are obtained by computing (G.m - d)**2

    returns: sum_sq, rms: sum of squared residuals, root mean square
    """
    sq = np.power(G@m - d,2)
    return np.sum(sq), np.sqrt(np.mean(sq))


def _m_to_station_error_matrix(m, d_indx, stations, nevt):
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


def _pairwise_delays_to_histories(dt, d_indx, stations, nm, evtdates, Cd):
    """
    Return timing error histories from inter-event pairwise timing delays, by
    solving the least squares problem station-wise.
    :param dt: Inter-event station timing delays, in s.
    :param d_indx:
    :param stations: list of stations
    :param nm:
    :param evtdates: list of event dates formatted as ...
    :param Cd: np.ndarray, Data covariance matrix
    :return:histories, dict
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
        timing_error, timing_cov = _solve_least_squares(Gsta[:, used_cols], dsta, dcov, Cm)
        utcdates = np.array([np.datetime64(datetime.utcfromtimestamp(ts))
                             for ts in evtdates[used_cols]
                             ])
        histories.update({staname: {'T_UTC_in_s': evtdates[used_cols],
                                    'delay_in_s': timing_error,
                                    'std_in_s': np.sqrt(np.diag(timing_cov)),
                                    'T_UTC': utcdates}})

    return histories


# def _pairwise_delays_to_histories(dt, d_indx, stations, nevt, evtdates, Cd):
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
#     timing_error, timing_cov = _solve_least_squares(Gsta[:, used_cols], dsta, dcov)
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


def _build_demeaned_delays(df,
                          evtnames,
                          min_sta_per_pair,
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
            _iter_over_event_pairs(df, evtnames, min_sta_per_pair):
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


def _write_timing_errors(outputdir, stations, histories):
    for sta in stations:
        filename = os.path.join(outputdir, f'timing_delays_{sta}.txt')
        with open(filename, 'wt', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(('T_UTC', 'T_UTC_in_s', 'delay_in_s', 'std_in_s'))
            writer.writerows(zip(histories[sta]['T_UTC'],
                                 histories[sta]['T_UTC_in_s'],
                                 histories[sta]['delay_in_s'],
                                 histories[sta]['std_in_s']))


def _write_residuals(outputdir, rms, sum_sq_res):
    filename = os.path.join(outputdir, 'residuals.txt')
    with open(filename, 'wt') as f:
         f.write(f'RMS = {rms}\n')
         f.write(f'SUM SQ. RES. = {sum_sq_res}')



