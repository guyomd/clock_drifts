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
        self.Cd = None
        self.Cm = None
        self.m = None
        self.sqres = None
        self.rms = None
        self.drifts = None

    def _build_inputs_for_inversion(self, vpvsratio, reference_stations):

        if self.dm.stations is None:
            print('!! Missing list of stations. Loading stations...')
            self.dm.list_stations()
        if self.dm.evtnames is None:
            print('!! Missing event names. Loading events...')
            self.dm.load_events()
        if self.dm.delays is None:
            print('!! Missing delays (!). Loading data...')
            self.dm.load_data()
        self.G, self.d, self.Cd, self.Cm, self.m_indx_s, self.m_indx_e = \
                _build_matrices(self.dm.delays,
                                vpvsratio,
                                self.dm.evtnames,
                                self.dm.records,
                                min_sta_per_pair=self.dm.min_sta_per_pair,
                                stations_wo_drift=reference_stations)


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

    def _convert_m_to_histories(self):
        """
        Return timing error histories from the array of pointwise, stationwise
        clock drift estimates g
        :param m: np.ndarray, array of pointwise, stationwise clock drift estimates, in s.
        :param cm: np.ndarray, a-posteriori covariance matrix on m
        :param station_records: dict, list of event recordings for each station

        :return:histories, dict
        """
        if (self.m is None) or (self.Cm is None):
            raise ValueError('Least squares inversion must be solved before caliing this method')
        ns = len(self.dm.records)
        stations = list(self.dm.records.keys())
        neps = []
        evtlist = []
        for v in self.dm.records.values():
            neps.append(len(v))

        self.drifts = dict()
        for s in stations:
            ista = stations.index(s)
            dates = []
            drift = []
            ims = np.nonzero(self.m_indx_s == ista)[0]
            dates = np.array([self.dm.evtdates[self.m_indx_e[im]] for im in ims])
            drift = np.array([self.m[im] for im in ims])

            isort = np.argsort(dates)
            utcdates = np.array([np.datetime64(datetime.utcfromtimestamp(d)) for d in dates])
            subcm = self.Cm[np.ix_(ims[isort], ims[isort])]
            assert subcm.shape[0] == len(isort)
            assert subcm.shape[1] == len(isort)
            self.drifts.update({s: {'T_UTC_in_s': dates[isort],
                                    'drift_in_s': drift[isort],
                                    'std_in_s': np.sqrt(np.diag(subcm)),
                                    'T_UTC': utcdates[isort]}})


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


    def write_outputs(self, outdir):
        """
        Write inversion results in a file

        :param outdir: str, Path to output directory
        """
        _write_timing_errors(outdir, self.dm.stations, self.drifts)
        _write_residuals(outdir, self.rms, self.sqres)

    def run(self, vpvsratio, reference_stations):
        print(f'\n>> [1/4] Build matrices for inversion')
        self.dm.filter_delays_on_evtnames()
        self._build_inputs_for_inversion(
            vpvsratio,
            reference_stations)
        print(f'\n>> [2/4] Run least-squares inversion')
        self._solve_least_squares()
        print(f'\n>> [3/4] Compute residuals')
        self._compute_residuals()
        print(f'         sum of square residuals: {self.sqres}')
        print(f'         root mean square: {self.rms}')
        print(f'\n>> [4/4] Reformat clock drift histories')
        self._convert_m_to_histories()
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

def _build_matrices(delays, vpvsratio, events, station_records, min_sta_per_pair=2,
                    verbose=False, stations_wo_drift=[]):
    """
    :param delays: Pandas.DataFrame, as formatted by load_data() method
    :param vpvsratio: float, vp/vs ratio
    :param events: list, event names
    :param station_records: dict, list of event recordings for each station (keys)
    :param min_sta_per_pair: int, minimum number of stations per event pair
    :param verbose: boolean, set verbosity
    :param stations_wo_drift: list of stations forced to have no timing errors
    :param add_closure_triplets: boolean, Flag specifying whether closure triplets should be appended to G.
    :returns:
    """
    if min_sta_per_pair < 2:
        raise ValueError('Minimum number of stations per event pair cannot be smaller than 2')

    d = []
    dcov = []
    g = []
    m_indx_s = []  # Station index for each element in m
    m_indx_e = []  # Event index for each element in m

    # Count stations:
    stations = list(station_records.keys())
    nsta = len(stations)

    # Count events per station:
    nev = len(events)
    nev_per_sta = []
    for (sta, rec) in station_records.items():
        ne = len(rec['evts'])
        nev_per_sta.append(ne)
        m_indx_s += [stations.index(sta)] * ne
        m_indx_e += [events.index(e) for e in rec['evts']]
    m_indx_s = np.array(m_indx_s)
    m_indx_e = np.array(m_indx_e)
    nm = sum(nev_per_sta)  # Total number of clock-drift time occurrences
    assert nm == len(m_indx_e)
    assert len(m_indx_s) == len(m_indx_e)
    print(f'Number of drift values: {nm}')

    # Filter delay table on minimum number of stations per pair:
    pairs = delays.groupby(['evt1', 'evt2']) \
                  .filter(lambda x: (len(x) > min_sta_per_pair))  # Pandas.DataFrame instance
    npairs = len(pairs)
    print(f'Event pairs (arrival-time delays) recorded by at least {min_sta_per_pair} stations: {npairs}')
    bar = ProgressBar(npairs)
    mcov = np.ones((nm,))/_QUASI_ZERO  # equiv. infinite a priori variance (undetermined)
    del_cnt = 0

    # For each event-pair arrival-time delay (at a single station), add a line in d and G:
    for grp_name, grp in pairs.groupby(['evt1', 'evt2']):
        evt1, evt2 = grp_name
        ie1 = events.index(evt1)
        ie2 = events.index(evt2)
        ista = [stations.index(s) for s in grp['station']]
        ns = len(ista) # number of stations in current group
        dtp_dm = (grp['dtP']-grp['dtP'].mean()).values
        dts_dm = (grp['dtS']-grp['dtS'].mean()).values
        dtpvar = grp['dtPvar'].values
        dtsvar = grp['dtPvar'].values
        pvar = (grp['dtPvar'] + grp['dtPvar'].sum()*np.power(1 / ns, 2)).values  # Variance on de-meaned P arrival time delays
        svar = (grp['dtSvar'] + grp['dtSvar'].sum()*np.power(1 / ns, 2)).values  # Variance on de-meaned S arrival time delays
        nex = len([s for s in grp['station'] if s in stations_wo_drift])

        for k in range(ns):
            # Add element to d (array of observations):
            d.append((dts_dm[k] - vpvsratio * dtp_dm[k]) / (1 - vpvsratio))
            dcov.append( (svar[k] + (vpvsratio**2)*pvar[k]) / (1 - vpvsratio)**2 )  # Data variance

            # Buildup G:
            g_line = np.zeros((nm,))
            im1 = np.where(np.logical_and(m_indx_e == ie1, m_indx_s == ista[k]))[0]
            im2 = np.where(np.logical_and(m_indx_e == ie2, m_indx_s == ista[k]))[0]
            g_line[im1] = 1
            g_line[im2] = -1
            # De-meaned arrival times: remove average differential drift from all stations:
            for k2 in range(ns):
                im1 = np.where(np.logical_and(m_indx_e == ie1, m_indx_s == ista[k2]))[0]
                im2 = np.where(np.logical_and(m_indx_e == ie2, m_indx_s == ista[k2]))[0]
                g_line[im1] -= 1 / ns
                g_line[im2] -= -1 / ns
            g.append(g_line)

        del_cnt += ns
        bar.update(del_cnt, title='Filling matrices G and d... ')

    # Add constraint on null initial clock drift value for each station:
    for s in stations:
        ista = stations.index(s)
        #ie0 = np.argsort(station_records[s]['dates'])[0]
        ie0 = np.argmin(station_records[s]['dates'])
        ie = events.index(station_records[s]['evts'][ie0])
        im = np.nonzero(np.logical_and(m_indx_e == ie, m_indx_s == ista))[0]
        g_line = np.zeros((nm,))
        g_line[im] = 1
        g.append(g_line)
        d.append(0.0)
        dcov.append(0.0)
        mcov[im] = _QUASI_ZERO
        # Add additional constraints on stations forced to have zero timing errors:
        if s in stations_wo_drift:
            enum_evts = [(indx, name) for indx, name in enumerate(station_records[s]['evts']) if indx != ie0]
            for _, evt in enum_evts:
                ie = events.index(evt)
                im = np.nonzero(np.logical_and(m_indx_e == ie, m_indx_s == ista))[0]
                g_line = np.zeros((nm,))
                g_line[im] = 1
                g.append(g_line)
                d.append(0.0)
                dcov.append(0.0)
                mcov[im] = _QUASI_ZERO

    g = np.array(g)
    d = np.array(d)
    Cd = np.diag(dcov)
    Cm = np.diag(mcov)
    print(f'Dimensions of array d: {d.shape}')
    print(f'Dimensions of matrix G: {g.shape}')
    return g, d, Cd, Cm, m_indx_s, m_indx_e


def _solve_least_squares(G, d, Cd, Cm):
    """
    Solves for m in the least squares problem "G.m = d", where Cd is the covariance matrix on d.
    :param G: matrix with NxM elements
    :param d: matrix with Nx1 elements: input data
    :param Cd: data covariance matrix, with NxN elements
    :param Cm: a priori covariance on model parameters
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


def _write_timing_errors(outputdir, stations, histories):
    for sta in stations:
        filename = os.path.join(outputdir, f'clock_drift_{sta}.txt')
        with open(filename, 'wt', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(('T_UTC', 'T_UTC_in_s', 'drift_in_s', 'std_in_s'))
            writer.writerows(zip(histories[sta]['T_UTC'],
                                 histories[sta]['T_UTC_in_s'],
                                 histories[sta]['drift_in_s'],
                                 histories[sta]['std_in_s']))


def _write_residuals(outputdir, rms, sum_sq_res):
    filename = os.path.join(outputdir, 'residuals.txt')
    with open(filename, 'wt') as f:
         f.write(f'RMS = {rms}\n')
         f.write(f'SUM SQ. RES. = {sum_sq_res}')



