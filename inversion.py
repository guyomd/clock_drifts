from clock_drifts import graphs
import os
from clock_drifts import data 
from clock_drifts import lib


def run(datafile, datatype, eventfile, vpvsratio,
        min_sta_per_evt, min_sta_per_pair,
        outputdir, make_plots=False, reference_stations=[],
        add_closure_relation=True):

    # Check existence of output directory:
    if not os.path.exists(outputdir):
        print(f'Directory {outputdir} does not exists.')
        print(f'Create directory {outputdir}.')
        os.makedirs(outputdir)

    # Print plotting status:
    print(f'>> Is plot option activated ? {make_plots}')

    # Load data:
    dm = data.DataManager(datafile, datatype)
    dm.load()
    print('>> Raw data analysis (file "{datafile}"):')
    dm.count_records_per_station()

    print(f'>> Input parameters:\n'+
          f'   Min. number of stations per evt = {min_sta_per_evt}\n' +
          f'   Min. number of stations per pair = {min_sta_per_pair}')
    dm.list_all_stations()
    print(f'   Reference stations (i.e. no drift):\n   {reference_stations}')


    print(f">> Identify events with at least {min_sta_per_evt} records")
    evtnames = dm.get_events_with_records(min_sta_per_evt=min_sta_per_evt)
    evtdates = dm.load_dates_from_file(eventfile)
    print(f'   {len(evtnames)} events ({len(evtdates)} dates) matching this criterion')

    cde = lib.ClockDriftEstimator(dm)
    drifts = cde.run(vpvsratio, 
                     reference_stations, 
                     min_sta_per_pair, 
                     add_closure_triplets=add_closure_relation)
    
    if make_plots:
        # Display relative timing errors:
        print(f'>> Make plots:')
        sem = lib._m_to_station_error_matrix(cde.m,
                                             cde.d_indx[:cde.ndel, :],
                                             dm.stations,
                                             len(dm.evtnames))
        for s in dm.stations:
            ax = graphs.plot_count_matrix(sem[s],
                                          colmap='RdBu_r',
                                          title=s,
                                          clabel='Relative timing error (s.)')
            graphs.save_plot(
                os.path.join(outputdir,
                             f'{s}_relative_timing_error_matrix.png'),
                             h=ax)

        # Plot time histories:
    ax = graphs.plot_clock_drift(drifts, dm.stations)
    graphs.save_plot(
        os.path.join(outputdir,
                     f'clock_drifts.png'),
                     h=ax)

    # Write to file:
    cde.write_outputs(outputdir)

    outputdict = {
                  'drifts': cde.drifts,
                  'stations': dm.stations,
                  'evtnames': dm.evtnames,
                  'evtdates': dm.evtdates,
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
    MIN_STA_PER_EVT = 2
    MIN_STA_PER_PAIR = 2
    OUTPUTDIR = './dataset/output/'
    VPVSRATIO = 1.732

    output = run(
            DATAFILE, 
            DATATYPE,
            EVENTFILE,
            VPVSRATIO,
            MIN_STA_PER_EVT,
            MIN_STA_PER_PAIR,
            OUTPUTDIR,
            make_plots=False,
            reference_stations=['STA00'])



