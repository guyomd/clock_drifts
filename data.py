import pandas as pd
import numpy as np

_QUASI_ZERO = 1E-6

class DataManager(object):
    def __init__(self, filename, datatype, verbose=False):
        self.filename = filename
        self.datatype = datatype
        self.verbose = verbose
        self.delays = None
        self.picks = None
        self.stations = None
        self.evtnames = None
        self.evtdates = None

    def load(self):
        print(f'>> Load data from file: {self.filename}')

        if self.datatype == "delays":
            self.delays = _load_data(
                self.filename,
                datatype=self.datatype,
                verbose=self.verbose)
        elif self.datatype == "pickings":
            self.delays, self.picks = _load_data(
                self.filename,
                datatype=self.datatype,
                verbose=self.verbose)
        else:
            raise ValueError(f'Unrecognized data type : {self.datatype}')

    def load_dates_from_file(self, eventfile, delim=';'):
        """
        Load event dates, and eventually event names from a CSV formatted as below:

        id; dates
        event_001; 1041379202.3554274
        event_002; 1041379201.6874202
        ...; ...

        :param eventfile: str, path to the file
        :param delim: str, CSV-format delimiter
        :returns dates: iterable list of event dates
        """
        events = pd.read_csv(
            eventfile,
            delimiter=delim,
            skipinitialspace=True,
            usecols=['id', 'dates']
        )
        #events['dates'].astype(float)
        if (self.evtnames is None):
            print('Event names have not been initiated in this instance. '+
                  f'Will load all events and dates from file "{eventfile}".')
            # Load all event names and dates from file:
            self.evtnames = events['id'].tolist()
            self.evtdates = events['dates'].values
        else:
            print('Note: will only load dates for events matching self.evtnames.')
            # Load only dates for events listed in evtnames:
            self.evtdates = events.loc[events['id'].isin(self.evtnames), 'dates'].values
        return self.evtdates

    def list_stations(self, verbose=True):
        self.stations = _list_available_stations(
            self.delays,
            verbose=verbose)
        return self.stations

    def count_stations_per_pair(self, nmin_sta_per_evt=2, nmin_sta_per_pair=0):
        if self.stations is not None:
            num_records, sta_records = \
                _count_stations_per_pair(
                    self.delays,
                    self.stations,
                    nstamin_per_evt=nmin_sta_per_evt,
                    nstamin_per_event_pair=nmin_sta_per_pair)
        else:
            raise ValueError("Stations list should be initialized first. Try self.list_stations() method.")
        return num_records, sta_records

    def get_event_names(self, nmin_sta_per_evt=0):
        self.evtnames = _get_event_names(
            self.delays,
            datatype=self.datatype,
            nmin_sta_per_evt=nmin_sta_per_evt)
        return self.evtnames

    def get_dates_from_pickings(self, nmin_sta_per_evt=0):
        if self.picks is not None:
            self.evtdates = _get_dates_from_pickings(
                self.picks,
                nstamin_per_evt=nmin_sta_per_evt)
        else:
            raise ValueError('Error. Input data was not loaded as arrival times.')
        return self.evtdates


# Functions:
def _load_pickings(pickfile, verbose=False):
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


def _load_delays(delayfile, verbose=False):
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


def _load_data(filename, datatype='delays', verbose=False):
    """
        Load input data (pickings or delays).

        :param filename: str, Path to file
        :param filetype: str, Type of data. Can be 'pickings' (for arrival time pickings) or 'delays' (for arrival time delays)
        :param verbose, bool, verbosity flag
        :return: pandas.DataFrame objects (1 if datatype="delays", 2 if datatype="pickings")
    """
    if datatype == 'delays':
        df = _load_delays(filename, verbose=verbose)
        return df
    elif datatype == 'pickings':
        p = _load_pickings(filename, verbose=verbose)
        df = _pickings2delays(p)
        return df, p
    else:
        raise ValueError(f'Unrecognized type of input data: "{datatype}"')


def _pickings2delays(df):
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


def _list_available_stations(df, verbose=False):
    """
    List uniquely stations available in Dataframe
    :param df: Dataframe of P & S picks, as loaded using load_data() method
    :param verbose: True/False
    :return: list of unique station names
    """
    list_sta = df['station'].unique()
    sta_sorted = np.sort(list_sta).tolist()
    if verbose:
        nsta = len(sta_sorted)
        print(f'List of stations available ({nsta}):\n{sta_sorted}')
    return sta_sorted


def _count_stations_per_pair(df, stations, nstamin_per_evt=2,
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


def _get_event_names_in_pickings(df, nstamin_per_evt=0):
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


def _get_event_names_in_delays(df, nstamin_per_evt=0):
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
    counts = list()
    for u in uniques:
        counts.append(evts.count(u))
    return uniques[np.array(counts) >= nstamin_per_evt].tolist()


def _get_event_names(df, datatype='pickings', nmin_sta_per_evt=0):
    if datatype == 'pickings':
        names = _get_event_names_in_pickings(df, nstamin_per_evt=nmin_sta_per_evt)
    elif datatype == 'delays':
        names = _get_event_names_in_delays(df, nstamin_per_evt=nmin_sta_per_evt)
    else:
        raise ValueError(f'Unrecognized data type: {datatype}')
    return names


def _get_dates_from_pickings(df, nstamin_per_evt=0):
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


def _count_tt_delay_pairs(df, evtnames, nstamin_per_event_pair):
    """
        Count the number of joint P & S arrival-time delay pairs in the dataset
        for pairs matching the minimum number of station per pair.

        :param df: Dataframe of P & S arrival time delays, as loaded using load_data() function
        :param evtnames: List of event names of interest
        :param nstamin_per_event_pair: Minimum number of stations with P and S delays per event pair
        :return: count: integer
        """
    count = 0
    for evt1, evt2, stations, dtp, dts, _, _ in _iter_over_event_pairs(df,
                                                                evtnames,
                                                                nstamin_per_event_pair):
        count += len(dtp)
    return count

def _iter_over_event_pairs(df, evtnames, nstamin):
    """
    Loop over all event pairs and return P & S traveltime delays
    for all pairs matching (i) event names and, (ii) minimum number
    of common stations given per pair.

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
                if np.all( [
                    np.bool(np.abs(row['dtP']) > 0.0), 
                    np.bool(np.abs(row['dtS']) > 0.0)] ):
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
