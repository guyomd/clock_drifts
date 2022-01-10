import lib, graphs
import os
from data import DataManager
from lib import ClockDriftEstimator


def run(datafile, datatype, eventfile, vpvsratio,
        nstamin_per_evt, nstamin_per_eventpair,
        outputdir, make_plots=False, reference_stations=[]):

    # Check existence of output directory:
    if not os.path.exists(outputdir):
        print(f'Directory {outputdir} does not exists.')
        print(f'Create directory {outputdir}.')
        os.makedirs(outputdir)

    # Print plotting status:
    print(f'>> Is plot option activated ? {make_plots}')

    # Load data:
    dm = DataManager(datafile, datatype, verbose=True)
    dm.load()
    dm.list_stations()
    print(f'>> Reference stations (i.e. no drift):\n{reference_stations}')

    print(f">> Get event names and dates")
    evtnames = dm.get_event_names(nmin_sta_per_evt=nstamin_per_evt)
    evtdates = dm.load_dates_from_file(eventfile)
    print(f'   total number of events: {len(evtnames)} ({len(evtdates)} dates)')

    cde = ClockDriftEstimator(dm)
    drifts = cde.run(vpvsratio, reference_stations, nstamin_per_eventpair)
    
    if make_plots:
        # Display relative timing errors:
        print(f'>> Convert relative drifts to clock drift histories')
        sem = lib._m_to_station_error_matrix(m,
                                             d_indx[:ndel, :],
                                             stations,
                                             len(evtnames))
        for s in stations:
            ax = graphs.plot_count_matrix(sem[s],
                                          colmap='RdBu_r',
                                          title=s,
                                          clabel='Relative timing error (s.)')
            graphs.save_plot(
                os.path.join(outputdir,
                             f'{s}_relative_timing_error_matrix.png'),
                             h=ax)

        # Plot time histories:
    ax = graphs.plot_clock_drift(drifts, stations)
    graphs.save_plot(
        os.path.join(outputdir,
                     f'clock_drifts.png'),
                     h=ax)

    # Write to file:
    cde.write_outputs(outputdir)

    outputdict = {
                  'teh': cde.histories,
                  'stations': dm.stations,
                  'm': cde.m,
                  'd': cde.d,
                  'd_indx': cde.d_indx,
                  'sum_sq_res': cde.sqres,
                  'rms': cde.rms
                 }

    return outputdict


if __name__ == "__main__":

    DATAFILE = './dataset/delays.txt' #./dataset/pickings.txt' 
    DATATYPE = 'delays' # 'pickings' 
    EVENTFILE = './dataset/events.txt'
    NSTAMIN_PER_EVT = 2
    NSTAMIN_PER_EVENTPAIR = 2
    OUTPUTDIR = './dataset/output/'
    VPVSRATIO = 1.732

    output = run(
            DATAFILE, 
            DATATYPE,
            EVENTFILE,
            VPVSRATIO,
            NSTAMIN_PER_EVT,
            NSTAMIN_PER_EVENTPAIR,
            OUTPUTDIR,
            make_plots=True,
            reference_stations=['STA00'])



