import lib, graphs
import os


class DataManager(object):
    def __init__(self, filename, datatype, verbose=True):
        self.filename = filename
        self.datatype = datatype
        self.verbose = verbose
        self.delays = None
        self.picks = Nonesns

    def load(self):
        print(f'>> Load data from file: {self.filename}')
        
        if self.datatype == "delays":
            self.delays = lib.load_data(
                    self.filename,
                    datatype=self.datatype, 
                    verbose=self.verbose)

        elif self.datatype == "pickings":
            self.delays, self.picks = lib.load_data(
                    self.filename,
                    datatype=self.datatype, 
                    verbose=self.verbose)
        
        else:
            raise ValueError(f'Unrecognized data type : {self.datatype}')


    def list_stations(self):
        # List stations:
        self.stations = lib.list_available_stations(
                self.delays, 
                verbose=verbose)
        return self.stations


   def count_stations_per_pair(self, nmin_sta_per_evt=2, nmin_sta_per_pair=0):
       num_records, sta_records = \
               lib.count_stations_per_pair(
                       self.delays,
                       self.stations,
                       nstamin_per_evt=nmin_sta_per_evt,
                       nstamin_per_event_pair=nmin_sta_per_pair)
        return num_records, sta_records


    def get_event_names(self, nmin_sta_per_evt=0):
        names = lib.get_event_names(
                self.delays, 
                datatype=self.datatype
                nmin_sta_per_evt=nmin_sta_per_evt)
        return names

 
    def get_dates_from_pickings(self, nmin_sta_per_evt=0):
        if self.picks is not None:
            dates = lib.get_dates_from_pickings(
                    self.picks,
                    nstamin_per_evt=nmin_sta_per_evt)
        else:
            raise ValueError('Error. Input data was not loaded as arrival times.')
        return dates




class ClockDriftEstimator(object):
    def __init__(self, dm: DataManager):
        self.dm = dm  # instance of DataManager object

    def _build_matrices_for_inversion(self):
        pass

    def _solve_least_squares(self):
        pass

    def _compute_residuals(self):
        pass

    def _pairwise_delays_to_histories(self):
         pass

    def write_output(self):
        #call lib.write_timing_errors() and lib.write_residuals()
        pass

    def run(self):
        build_matrices_for_inversion()
        solve_least_squares()
        compute_residuals()
        pairwise_delays_to_histories()
        pass
        # Return Python dict as output


def run(datafile, datatype, vpvsratio, nstamin_per_evt, nstamin_per_eventpair,
         min_delay, outputdir, make_plots=False, stations_wo_error=[],
         filetype="pickings"):

    # Check existence of output directory:
    if not os.path.exists(outputdir):
        print(f'Directory {outputdir} does not exists.')
        print(f'Create directory {outputdir}.')
        os.makedirs(outputdir)

    # Print plotting status:
    print(f'>> Plot option activated: {make_plots}')


    # Load data:
    dm = DataManager(datafile, datatype, verbose=True)
    dm.load()

    # List all stations:
    dm.list_stations()
    print(f'>> Reference stations (i.e. account only for picking uncertainty, no timing delay):\n{stations_wo_error}')

    print(f'>> Count number of records per event pair')
    nb_rec, sta_rec = dm.count_stations_per_pair(
            nmin_sta_per_evt=nstamin_per_evt, 
            nmin_sta_per_pair=nstamin_per_eventpair)

    # Build matrices for the inversion of station timing errors:
    print(f">> Get event names")
    evtnames = dm.get_event_names(nmin_sta_per_evt=nstamin_per_evt)
    nevt = len(evtnames)
    print(f'\n\nTotal number of events: {nevt}')

    if datatype == 'pickings':
        print(f">> Get event dates")
        evtdates = dm.get_dates_from_pickings(nmin_sta_per_evt=nstamin_per_evt)
        print(f'\n\nTotal number of event dates: {len(evtdates)}')


##### TO BE CONTINUED (MODIFICATIONS TO USE DELAYS INSTEAD OF PICKINGS) #####
    print(f'>> Build matrices for inversion')
    G, d, d_indx, Cd, Cm, ndel = lib.build_matrices_for_inversion(df,
                                         evtnames,
                                         stations,
                                         vpvsratio,
                                         nstamin_per_event_pair=nstamin_per_eventpair,
                                         min_delay=min_delay,
                                         stations_wo_error=stations_wo_error)

    # Run inversion:
    print(f'>> Run inversion')
    m, Cm = lib.solve_least_squares(G, d, Cd, Cm)

    # Compute residuals:
    sqres, rms = lib.compute_residuals(G, d, m)
    print(f'>> sum of square residuals: {sqres}')
    print(f'>> root mean square: {rms}')

    # Display relative timing errors:
    if make_plots:
        print(f'>> Convert relative delays to timing delay histories')
        sem = lib.m_to_station_error_matrix(m,
                                            d_indx[:ndel, :],
                                            stations,
                                            nevt)
        for s in stations:
            ax = graphs.plot_count_matrix(sem[s],
                                          colmap='RdBu_r',
                                          title=s,
                                          clabel='Relative timing error (s.)')
            graphs.save_plot(os.path.join(outputdir, f'{s}_relative_timing_error_matrix.png'),
                             h=ax)

    # Convert relative timing errors to timing error histories:
    timing_error_histories = lib.pairwise_delays_to_histories(m,
                                                              d_indx[:ndel, :],
                                                              stations,
                                                              nevt,
                                                              evtdates,
                                                              Cm)

    if make_plots:
        # Plot time histories:
        ax = graphs.plot_timing_error_histories(timing_error_histories,
                                                stations)
        graphs.save_plot(os.path.join(outputdir, f'timing_error_histories.png'),
                         h=ax)

    # Write timing errors histories:
    lib.write_timing_errors(outputdir, stations, timing_error_histories)
    lib.write_residuals(outputdir, rms, sqres)

    outputdict = {
                  'teh': timing_error_histories,
                  'stations': stations,
                  'm': m,
                  'd': d,
                  'd_indx': d_indx,
                  'sum_sq_res': sqres,
                  'rms': rms
                 }

    return outputdict


if __name__ == "__main__":

    datafile = './pickings.txt' 
    datatype = 'pickings' # "delays"
    NSTAMIN_PER_EVT = 2
    NSTAMIN_PER_EVENTPAIR = 2
    MIN_DELAY = 0.0  # in s.
    OUTPUTDIR = './output/'
    VPVSRATIO = 1.732

    output = run(
            datafile, 
            datatype,
            VPVSRATIO,
            NSTAMIN_PER_EVT,
            NSTAMIN_PER_EVENTPAIR,
            MIN_DELAY,
            OUTPUTDIR,
            make_plots=True,
            stations_wo_error=['CLAN'])



