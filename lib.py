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

        self.G, self.d, self.Cd, self.Cm, self.ndel = \
                _build_matrices(self.dm.delays,
                                vpvsratio,
                                self.dm.records,
                                min_sta_per_pair=self.dm.min_sta_per_pair,
                                stations_wo_drift=reference_stations,
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

    def _convert_m_to_histories(self, station_records):
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
        ns = len(stations_records)
        stations = list(stations_records.keys())
        neps = []
        evtlist = []
        for v in stations_records.values():
            neps.append(len(v))

        self.histories = dict()
        for s in stations:
            ista = stations.index(s)
            ix = sum(neps[:ista])
            it = np.argsort(station_records[s]['dates'])

            utcdates = np.array([np.datetime64(datetime.utcfromtimestamp(ts))
                                 for ts in station_records[s]['dates'][it]
                                 ])
            subcm = self.cm(np.ix_(ix + it, ix + it))
            assert subcm.shape[0] == len(ix + it)
            assert subcm.shape[1] == len(ix + it)
            histories.update({staname: {'T_UTC_in_s': station_records[s]['dates'][it],
                                        'delay_in_s': self.m[ix + it],
                                        'std_in_s': np.sqrt(np.diag(subcm)),
                                        'T_UTC': utcdates}})
        return histories


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
        self.dm.filter_delays_on_evtnames()
        self._build_inputs_for_inversion(
            vpvsratio,
            reference_stations,
            add_closure_triplets=add_closure_triplets)
        print(f'\n>> [2/4] Run inversion of relative drifts')
        self._convert_m_to_histories(m, cm, station_records)
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

def _build_matrices(delays, vpvsratio, station_records, min_sta_per_pair=2,
                    verbose=False, stations_wo_drift=[], add_closure_triplets=True):
    """
    :param delays: Pandas.DataFrame, as formatted by load_data() method
    :param evtnames: list of event (names) used in the inversion
    :param stations_used: list of stations used in the inversion
    :param vpvsratio: float, vp/vs ratio
    :param station_records: dict, list of event recordings for each station (keys)
    :param min_sta_per_pair: int, minimum number of stations per event pair
    :param verbose: boolean, set verbosity
    :param stations_wo_drift: list of stations forced to have no timing errors
    :param add_closure_triplets: boolean, Flag specifying whether closure triplets should be appended to G.
    :return:
    """
    if min_sta_per_pair < 2:
        raise ValueError('Minimum number of stations per event pair cannot be smaller than 2')

    d = []
    dcov = []
    g = []
   # Count stations:
    ns = len(stations_records)
    stations = list(stations_records.keys())

    # Count events:
    neps = []
    evtlist = []
    for d in stations_records.values():
        neps.append(len(d['evts']))
        evtlist += d['evts']
    nt = sum(neps)  # Total number of clock-drift time occurrences

    # Filter delay table on minimum number of stations per pair:
    pairs = delays.groupby(['evt1', 'evt2']) \
                  .filter(lambda x: (len(x) > min_sta_per_pair))  # Pandas.DataFrame instance
    nm = len(pairs)
    print(f'Event pairs (arrival-time delays) recorded by at least {min_sta_per_pair} stations: {nm}')
    bar = ProgressBar(nm)
    mcov = np.ones((nt,))/_QUASI_ZERO  # equiv. infinite a priori variance (undetermined)
    del_cnt = 0

    # For each event-pair arrival-time delay (at a single station), add a line in d and G:
    for grp_name, grp in pairs.groupby(['evt1', 'evt2']):
        evt1, evt2 = grp_name
        dtp_dm = (grp['dtP']-grp['dtP'].mean()).values
        dts_dm = (grp['dtS']-grp['dtS'].mean()).values
        dtpvar = grp['dtPvar'].values
        dtsvar = grp['dtPvar'].values
        n12 = len(grp['dtP'])  # Number of delays for the current pair: (evt1, evt2)
        pvar = (grp['dtPvar'] + grp['dtPvar'].sum()*np.power(1 / n12, 2)).values  # Variance on de-meaned P arrival time delays
        svar = (grp['dtSvar'] + grp['dtSvar'].sum()*np.power(1 / n12, 2)).values  # Variance on de-meaned S arrival time delays
        nex = len([s for s in grp['station'] if s in stations_wo_drift])

        ista = [stations.index(s) for s in grp['station']]
        ns = len(ista)

        for k in range(ns):
            # Add element to d (array of observations):
            d.append((dts_dm[k] - vpvsratio * dtp_dm[k]) / (1 - vpvsratio))
            dcov.append( (svar[k] + (vpvsratio**2)*pvar[k]) / (1 - vpvsratio)**2 )  # Data variance

            # Buildup G:
            g_line = np.zeros((nt,))
            ix = sum(neps[:ista[k]])
            ix1 = ix + station_records[stations[ista[k]]]['evts'].index(evt1)
            ix2 = ix + station_records[stations[ista[k]]]['evts'].index(evt2)
            g_line[ix1] = 1 - 1 / n12
            g_line[ix2] = - (1 - 1 / n12)
            # De-meaned timing error: remove average timing error for all other stations recording this event pair:
            for k2 in (j for j in range(ns) if j!=k):
                iy = sum(neps[:ista[k2]])
                iy1 = iy + station_records[stations[ista[k2]]]['evts'].index(evt1)
                iy2 = iy + station_records[stations[ista[k2]]]['evts'].index(evt2)
                g_line[iy1] = -1 / n12
                g_line[iy2] = 1 / n12
            g.append(g_line)

            # Eventually, add constraint on stations forced to have zero timing errors (+/- picking errors):
            if stations[ista[k]] in stations_wo_drift:
                d.append(0.0)
                dcov.append(0.0)  # Data variance
                g_line = np.zeros((nt,))
                g_line[ix1] = 1
                g_line[ix2] = -1
                g.append(g_line)

        del_cnt += n12
        bar.update(del_cnt, title='Filling matrices G and d... ')

    # Add constraint on null initial clock drift value for each station:
    for s in stations:
        ista = stations.index(s)
        ix = sum(neps[:ista])
        i0 = np.argsort(station_records[s]['dates'])[0]
        g_line = np.zeros((nt,))
        g_line[ix+i0] = 1
        g.append(g_line)
        d.append(0.0)
        dcov.append(0.0)

    g = np.array(g)
    d = np.array(d)
    Cd = np.diag(dcov)
    Cm = np.diag(mcov)
    print(f'Dimensions of array d: {d.shape}')
    print(f'Dimensions of matrix G: {g.shape}')
    return g, d, Cd, Cm, ndel


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



