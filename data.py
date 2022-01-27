import pandas as pd
import numpy as np

_QUASI_ZERO = 1E-6

class DataManager(object):
    def __init__(self, filename, datatype, min_sta_per_evt=0, min_sta_per_pair=0, verbose=False):
        self.filename = filename
        self.datatype = datatype
        self.verbose = verbose
        self.delays = None
        self.picks = None
        self.stations = None
        self.evtnames = None
        self.evtdates = None
        self.min_sta_per_evt = min_sta_per_evt
        self.min_sta_per_pair = min_sta_per_pair

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
            print('   Event names not previously initiated. '+
                  f'  Load dates for all events in "{eventfile}".')
            # Load all event names and dates from file:
            self.evtnames = events['id'].tolist()
            self.evtdates = events['dates'].values
        else:
            # Load only dates for events listed in evtnames:
            self.evtdates = events.loc[events['id'].isin(self.evtnames), 'dates'].values
        return self.evtdates


    def list_all_stations(self, verbose=True):
        self.stations = _list_available_stations(
            self.delays,
            verbose=verbose)
        return self.stations


    def count_stations_per_pair(self):
        if self.stations is not None:
            num_records, sta_records = \
                _count_stations_per_pair(self.delays, self.stations)
        else:
            raise ValueError("Stations list should be initialized first. Try self.list_stations() method.")
        return num_records, sta_records


    def count_records_per_station(self):
        records = dict()
        print(f'number of records per station:')
        for sta, grp in self.delays.groupby('station'):
            uniq_evts = np.unique( np.append( grp['evt1'].unique(), grp['evt2'].unique() ) )
            records.update({sta: uniq_evts})
        for sta in records.keys():
            print(f'{sta}: {len(records[sta])}')
        return records


    def get_events_with_records(self):
        if self.datatype == 'pickings':
            self.evtnames = _get_event_names_in_pickings(
                    self.picks,
                    min_sta_per_evt=self.min_sta_per_evt)
        elif self.datatype == 'delays':
            self.evtnames = _get_event_names_in_delays(
                    self.delays,
                    min_sta_per_evt=self.min_sta_per_evt)
        else:
            raise ValueError(f'Unrecognized data type: {iself.datatype}')
        return self.evtnames


    def get_dates_from_pickings(self):
        if self.picks is not None:
            self.evtdates = _get_dates_from_pickings(
                self.picks,
                min_sta_per_evt=self.min_sta_per_evt)
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
        return df[ (df['dtP'] > 0.0) & (df['dtS'] > 0.0) ]

    elif datatype == 'pickings':
        p = _load_pickings(filename, verbose=verbose)
        if verbose:
            print('  Compoute inter-event arrival-time delays')
        df = _pickings2delays(p)
        return df[ (df['dtP'] > 0.0) & (df['dtS'] > 0.0) ], p
    
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
                sta1 = df.loc[j1, 'station']
                if sta1 in df.loc[indexes2, 'station'].values:
                    j2 = df.loc[indexes2, 'station'].tolist().index(sta1)
                    j2 = indexes2[j2]
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
                            delays['dtP'].append(df.loc[j1, 'tP'] - df.loc[j2, 'tP'])
                            delays['dtPerr'].append(df.loc[j1, 'tPerr'] + df.loc[j2, 'tPerr'])  # std. deviation
                            delays['dtPvar'].append( (df.loc[j1, 'tPerr'] + df.loc[j2, 'tPerr'])**2 )  # variance
                        else:
                            delays['dtP'].append(_QUASI_ZERO)
                            delays['dtPerr'].append(_QUASI_ZERO)
                            delays['dtPvar'].append(_QUASI_ZERO)

                        if np.all(s_arr):
                            delays['dtS'].append(df.loc[j1, 'tS'] - df.loc[j2, 'tS'])
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
        print(f'   {nsta} stations available in the dataset:\n   {sta_sorted}')
    return sta_sorted


def _count_stations_per_pair(df, stations):
    evtnames = np.unique(
        np.append(pd.unique(df['evt1']).values,
                  pd.unique(df['evt1']).values)
    )

    nevt = len(evtnames)
    print(f'   {len(evtnames)} events have at least {min_sta_per_evt} records')

    # Initialize  matrix of station counts per event pair:
    num_records = np.zeros((nevt, nevt))
    evt_records = {s: np.zeros((nevt, nevt))
                   for s in stations}

    # Count station for each pair:
    for name, grp in df.groupby(['evt1', 'evt2']):
        evt1, evt2 = name
        ie_1 = evtnames.index(evt1)
        ie_2 = evtnames.index(evt2)
        siz = grp.size()
        num_records[ie_1, ie_2] = siz
        num_records[ie_2, ie_1] = siz
        for s in grp['station']:
            evt_records[s][ie_1, ie_2] = 1
            evt_records[s][ie_2, ie_1] = 1

    return num_records, evt_records


def _get_event_names_in_pickings(df, min_sta_per_evt=0):
    """
    Return a list of event names
    :param df: Dataframe of P & S picks, as loaded using load_pickings() method
    :param min_sta_per_evt:
    :return:
    """
    evtnames = [name
                for name, subdf in df.groupby('event')
                if len(subdf) >= min_sta_per_evt]
    return evtnames


def _get_event_names_in_delays(df, min_sta_per_evt=0):
    """
    Return a list of event names
    :param df: Dataframe of P & S arrival time delays
    :param min_sta_per_evt:
    :return:
    """
    pairs = []
    for _, row in df.iterrows():
        if (np.abs(row['dtP'])>0) and (np.abs(row['dtS'])>0):
            pairs.append(f'{row["evt1"]}-#-{row["station"]}')  # "event-station" pair
            pairs.append(f'{row["evt2"]}-#-{row["station"]}')
    uniq_pairs = np.unique(pairs)
    evts = list()
    for p in uniq_pairs:
        evts.append(p.split('-#-')[0])
    uniq_evts = np.unique(evts)
    counts = list()
    for u in uniq_evts:
        counts.append(evts.count(u))
    return uniq_evts[np.array(counts) >= min_sta_per_evt].tolist()  


def _get_dates_from_pickings(df, min_sta_per_evt=0):
    """
    Return a numpy array of event dates
    :param df: Dataframe of P & S picks, as loaded using load_pickings() method
    :param min_sta_per_evt:
    :return:
    """
    evtdates = np.array([subdf.loc[subdf['tP'] > 0, 'tP'].min()
                         for name, subdf in df.groupby('event')
                         if len(subdf) >= min_sta_per_evt])
    return evtdates


