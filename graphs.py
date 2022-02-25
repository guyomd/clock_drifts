import os   
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib.cm as cmx

def save_plot(filename, h=None, size=(8,6), keep_opened=False, tight=False, **kwargs):
    if isinstance(h, plt.Axes):
        h = h.get_figure()  # Eventually, retrieve parent Figure handle
    elif isinstance(h, plt.Figure):
        pass
    elif h is None:
        print('INFO: save_plot() method will save the current figure')
        h = plt.gcf()
    plt.figure(h.number)   # Set as current figure
    h.set_size_inches(size[0], size[1])  # size = (width, height)
    if tight:
        kwargs.update({'bbox_inches': 'tight'})
    else:
        kwargs.update({'bbox_inches': None})
    plt.savefig(filename, dpi=150, pad_inches=0.05, **kwargs)
    print(f'Figure saved in {filename}')
    if not keep_opened:
        plt.close(h.number)


def histogram_num_records_per_event(df_picks, title=None, yscale="log"):
    """
    Draw a histogram of number of records per event
    :param df: Dataframe of P & S picks, as loaded using lib.load_pickings() method
    :param title:
    :return:
    """
    grp = df_picks.groupby('evt1')
    print(f'Number of events: {len([1 for evt in grp])}')

    grp.size().hist()
    plt.xlabel('Number of records')
    plt.ylabel('Number of events')
    plt.yscale(yscale)
    plt.title(title)
    ax = plt.gca()
    return ax


def plot_count_matrix(count_matrix, colmap='terrain_r',
                      title=None, clabel=None,
                      add_colorbar=True):
    # Display matrix of counts:
    plt.matshow(count_matrix, cmap=colmap)
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.title(title)
    ax = plt.gca()
    if add_colorbar:
        nsmax = int(np.nanmax(count_matrix))
        cbar = plt.colorbar()
        cbar.set_label(clabel)
        #cbar.set_ticks([x for x in range(nsmax)])
    return ax


def plot_clock_drift(histories, stations, cmap='tab20', time_converter=None, add_uncertainties=False):
    plt.figure()
    cNorm = colors.Normalize(vmin=0, vmax=len(stations)-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=plt.get_cmap(cmap))
    for s in histories.keys():
        ista = stations.index(s)
        colorVal = scalarMap.to_rgba(ista)
        if len(histories[s]['T_UTC_in_s']) > 0:
            if (time_converter is not None) and (s in list(time_converter.keys())): 
                # Convert time in second using the function provided in time_converter[s]:
                time = time_converter[s](histories[s]['T_UTC_in_s'])
            else: 
                # Use ISO datetime format:
                time = np.array(histories[s]['T_UTC'])
            plt.plot(time, histories[s]['drift_in_s'], color=colorVal, label=s)
            if add_uncertainties:
                plt.plot(time, histories[s]['drift_in_s'] - histories[s]['std_in_s'], color=colorVal, lw=0.5)
                plt.plot(time, histories[s]['drift_in_s'] + histories[s]['std_in_s'], color=colorVal, lw=0.5)

    plt.xlabel('Time')
    plt.ylabel('Timing delay [s]')
    ax = plt.gca()

    # Major ticks every month.
    fmt_month = mdates.MonthLocator()
    ax.xaxis.set_major_locator(fmt_month)

    # Minor ticks every week.
    fmt_week = mdates.WeekdayLocator(byweekday=1, interval=1)
    ax.xaxis.set_minor_locator(fmt_week)

    # Text in the x axis will be displayed in 'YYYY-mm' format.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    datemin = np.datetime64('2002-12-01')
    datemax = np.datetime64('2003-07-01')
    ax.set_xlim(datemin, datemax)

    # Format the coords message box, i.e. the numbers displayed as the cursor moves
    # across the axes within the interactive GUI.
    ax.format_xdata = mdates.DateFormatter('%Y-%m')
    ax.format_ydata = lambda x: f'{x:.2f} s.'  # Format the timing error.
    ax.grid(True)

    plt.gcf().autofmt_xdate()
    plt.legend()
    return ax


def plot_demeaned_delays(dtp, dts):
    plt.figure()
    plt.plot(dtp, dts, '.', fillstyle='full')
    ax = plt.gca()
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    # Add 1:1 and 1.732:1 trend lines:
    bounds = np.array([min(xl[0], yl[0]), max(xl[1], yl[1])])
    plt.plot(bounds, np.sqrt(3) * bounds, '--k', label='slope=1.732')
    plt.plot(bounds, bounds, ':k', label='slope=1.0')
    plt.xlabel('$\\tilde{\delta t_p}$ (s.)')
    plt.ylabel('$\\tilde{\delta t_s}$ (s.)')
    plt.grid(True, which='both', linestyle=':', lw=1)
    plt.legend()
    return ax


def plot_drifts(outputdir, sharey=False, show=True, figsize=(10,14)):
    """
    Displays clock drift histories, with subplots distributed in 2 columns, one station per subplot.

    :param outputdir: str, path to the directory which contains files "clock_drift_*.txt"
    :param sharey: bool, If True, share the same Y-axes limits among all subplot. Default: False.
    
    :returns f: Matplotlib.Pyplot.Figure instance
    """
    dfs = dict()
    for filename in glob.glob(os.path.join(outputdir,"clock_drift_*.txt")):
        sta = filename.split('_')[-1].split('.')[0]
        dfs.update({sta: pd.read_csv(filename, sep=';')})

    ns = len(dfs.keys())
    stations = list(dfs.keys())
    f, ax = plt.subplots(
            nrows=int(np.ceil(ns/2)), 
            ncols=2, 
            sharex=True, 
            sharey=sharey,
            figsize=figsize)
    ax = ax.flatten()
    all_ti = np.array([np.datetime64(s) for key in dfs.keys() for s in dfs[key]['T_UTC'].values])
    xmin = np.min(all_ti)
    xmax = np.max(all_ti)
    ymin = 0
    ymax = 0
    for i in range(ns):
        x = np.array([np.datetime64(s) for s in dfs[stations[i]]['T_UTC'].values])
        y = dfs[stations[i]]['drift_in_s'].values
        dy = dfs[stations[i]]['std_in_s'].values
        if y.min()<ymin: ymin=y.min()
        if y.max()>ymax: ymax=y.max()
        ax[i].plot_date([xmin, xmax], [0, 0], ':k', lw=0.5, ydate=False)
        ax[i].plot_date(x, y-dy, 'k--', ydate=False, lw=0.5)
        ax[i].plot_date(x, y+dy, 'k--', ydate=False, lw=0.5)
        ax[i].plot_date(x, y, 'k.-', markersize=6, label=stations[i], ydate=False, lw=1.5)
        if not sharey:
            ylimit = np.max(np.abs(ax[i].get_ylim()))
            ylimit = max([ylimit, 1])
            ax[i].set_ylim([-ylimit, ylimit])
        ax[i].set_ylabel('drift [s]')
        ax[i].legend()

    ymin = np.min([ymin, -ymax])
    ymax = np.max([-ymin, ymax])
    ax[0].set_xlim([xmin, xmax])
    if sharey:
        ax[0].set_ylim([ymin, ymax])
    plt.gcf().autofmt_xdate()
    if show:
        plt.show()
    return f


