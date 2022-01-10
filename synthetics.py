#!/usr/bin/env python

# External librairies:
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.dates as mdates
import os
import pandas as pd
from datetime import datetime
from copy import deepcopy
import sys

# Local imports:
from graphs import save_plot, plot_clock_drift
from lib import _load_pickings, _iter_over_event_pairs


MAKE_PLOTS = False
OUTPUTDIR = sys.argv[1] 
if len(sys.argv)>2:
    VPOVERVS = float(sys.argv[2])
else:
    VPOVERVS = 1.732

if len(sys.argv)>3:
    STDERR_TP = float(sys.argv[3])
else:
    STDERR_TP = 0.0

if len(sys.argv)>4:
    if sys.argv[4] == "RANDOM_TT_PERTURB":
        seed_tt = np.random.randint(100000000)
else:
    seed_tt = 23


OUTPUT_PICKFILE = os.path.join(OUTPUTDIR, 'dataset.txt')
OUTPUT_EVENTFILE = os.path.join(OUTPUTDIR, 'events.txt')
OUTPUT_DELAYFILE = os.path.join(OUTPUTDIR, 'delays.txt')
REFERENCE_PICKFILE = os.path.join(OUTPUTDIR, 'reference.txt')

# Dataset parameters:
seed = 23
vp = 3.0 # km/s
vpvsratio = VPOVERVS
tPerr = STDERR_TP  # std. dev. of P-wave picking errors
tSerr = tPerr*2  # std. dev. of S-wave picking errors

# Major ticks every month.
fmt_month = mdates.MonthLocator()

# # 1- Generate synthetic events, stations and traveltimes

# ## a- Set geometry for events and stations
rng = np.random.default_rng(seed)
if seed_tt != seed:
    rng_tt = np.random.default_rng(seed_tt)
else:
    rng_tt = rng

# Stations:
nsta = 15
sx_bounds = [-10.0, 10.0]
sy_bounds = [-10.0, 10.0]

#sx = rng.uniform(low=sx_bounds[0], high=sx_bounds[1], size=nsta)
#sy = rng.uniform(low=sy_bounds[0], high=sy_bounds[1], size=nsta)
sx = rng.normal(0, 0.2*(sx_bounds[1]-sx_bounds[0]), size=nsta)
sy = rng.normal(0, 0.2*(sy_bounds[1]-sy_bounds[0]), size=nsta)
sz = np.zeros((nsta,))  # Surface
stanames = [f'STA{i:02d}' for i in range(nsta)]

# Event cluster: 
nev = 20
ex_bounds = [-1.0, 1.0]
ey_bounds = [-1.0, 1.0]
ez_bounds = [0.5, 1.5]
ex = rng.uniform(low=ex_bounds[0], high=ex_bounds[1], size=nev)
ey = rng.uniform(low=ey_bounds[0], high=ey_bounds[1], size=nev)
ez = rng.uniform(low=ez_bounds[0], high=ez_bounds[1], size=nev)
evtnames = [f'event_{i:03d}' for i in range(nev)]

# Origin times (in s.):
tmin = np.datetime64("2003-01-01 00:00:00") - np.datetime64('1970-01-01')
tmin /= np.timedelta64(1, 's')
# TI between 0 and 120 days after TMIN
dt = rng.integers(low=0,high=120*24*60*60, size=nev)
ti = tmin + np.linspace(0,120*24*60*60,nev)

if MAKE_PLOTS:
    # Plot map:
    fig = plt.figure()
    plt.plot(ex, ey, 'xk')
    plt.plot(sx, sy, 'vr', markerfacecolor='r')
    plt.xlabel('X [km]')
    plt.ylabel('Y [km]')
    plt.grid(True, which='both')
    save_plot(os.path.join(OUTPUTDIR, 'map.png'), h=fig, size=(8,6))
    
# ## b- Set synthetic timing delays and update theoretical pickings:
ti_utc = np.array([np.datetime64(datetime.utcfromtimestamp(i)) for i in ti])
ERR = dict()
for s in stanames:
    ERR.update({s:
                    {'ti': ti,
                     'ti_utc': ti_utc,
                     'delay_s': np.zeros(len(ti),)}
                })

ERR.update(
    {'STA02':
       {'ti': ti,
        'ti_utc': ti_utc,
#        'delay_s': np.linspace(0, 100, len(ti))
        'delay_s': 10*np.sin( 2 * np.pi * np.linspace(0, 2, len(ti)) )
       },
     'STA05':
       {'ti': ti,
        'ti_utc': ti_utc,
        'delay_s': np.concatenate((np.linspace(0, -15, int(np.floor(nev/2))),
                                   np.linspace(20,0,len(ti)-int(np.floor(nev/2)))))
       },
     'STA07':
       {'ti': ti,
        'ti_utc': ti_utc,
        'delay_s': np.linspace(0, 30, len(ti))
       }
    })

# ## c- Compute synthetic traveltimes for P- and S-waves:

# Compute traveltimes:
vs = vp/vpvsratio
pickings = {'event': [], 
            'station': [],
            'channel': [],
            'tP': [],
            'tS': [],
            'tPerr': [],
            'tSerr': [],
            'timing_delay': [],
            'ti': [],
            'ti_utc': [],
            'evt2sta_dist': []
           }
for i in  range(nev):
    for j in range(nsta):
        d = np.sqrt( (ex[i]-sx[j])**2 + (ey[i]-sy[j])**2 + (ez[i]-sz[j])**2 )  # in km
        pickings['event'].append(evtnames[i])
        pickings['station'].append(stanames[j])
        pickings['channel'].append('HHZ')
        pickings['tP'].append(ti[i]+d/vp)
        pickings['tS'].append(ti[i]+d/vs)
        pickings['tPerr'].append(tPerr)
        pickings['tSerr'].append(tSerr)
        pickings['timing_delay'].append(ERR[stanames[j]]['delay_s'][i])  # Clock drift
        pickings['ti'].append(ti[i])
        pickings['ti_utc'].append(ti_utc[i])
        pickings['evt2sta_dist'].append(d)
pk = pd.DataFrame(pickings)

pk0 = deepcopy(pk)  # Backup theoretical error-free pickings
pk0['tPerr'] = 0.0
pk0['tSerr'] = 0.0
pk0.to_csv(REFERENCE_PICKFILE, sep=';', index=False)  # Export to file
print(f'Reference pickings saved in file "{REFERENCE_PICKFILE}"')

# Update pickings with timing delays and picking errors:
for sta in ERR.keys():
    s = str(sta)
    ii = pk['station']==s
    pk.loc[ii, 'tP'] += np.array(ERR[sta]['delay_s']) + rng_tt.normal(0, tPerr, size=len(pk.loc[ii, 'tP']))
    pk.loc[ii, 'tS'] += np.array(ERR[sta]['delay_s']) + rng_tt.normal(0, tSerr, size=len(pk.loc[ii, 'tS']))

# EXPORT SYNTHETIC DATASET TO FILES
# Event info:
with open(OUTPUT_EVENTFILE, 'wt') as f:
    f.write('id; dates')
    for i in range(nev):
        f.write(f'{evtnames[i]}; {ti[i]}')

# Pickings:
pk.drop(columns=['timing_delay', 'ti', 'ti_utc', 'evt2sta_dist'])
pk.to_csv(OUTPUT_PICKFILE, sep=';', index=False)
print(f'Synthetic dataset saved in file "{OUTPUT_PICKFILE}"')

# Delays:
with open(OUTPUT_DELAYFILE, 'wt') as fd:
    fd.write('evt1; evt2; station; channel; dtP; dtS; dtPerr; dtSerr')
    lines = []
    for i1 in range(nev):
        for i2 in range (i,nev):
            for k in range(nsta):
                evt1 = evtnames[i1]
                evt2 = evtnames[i2]
                sta = stanames[k]
                dtp = pk.loc[ (pk['station']==sta) & (pk['event']==evt1), 'tP' ] \
                    - pk.loc[ (pk['station']==sta) & (pk['event']==evt2), 'tP' ]
                dts = pk.loc[(pk['station'] == sta) & (pk['event'] == evt1), 'tS'] \
                    - pk.loc[(pk['station'] == sta) & (pk['event'] == evt2), 'tS']
                lines.append(f'{evt1}; {evt2}; {sta}; HHZ; {dtp}; {dts}; {2*tPerr}; {2*tSerr}')
    fd.writelines(lines)




if MAKE_PLOTS:
    # Plot synthetic timing delays:
    plt.figure()
    plt.plot(ERR['STA02']['ti_utc'], 0*ERR['STA02']['ti'], ls=':', color='k')
    for sta in ERR.keys():
        s = str(sta)
        ii = pk['station']==s
          
        t0 = pk0.loc[ii,'tP']
        t1 = pk.loc[ii,'tP']
        plt.plot(ERR[sta]['ti_utc'], t1-t0, label=s)
        plt.legend()

    plt.xlabel('Hypothetical time [s]')
    plt.ylabel('Station timing delay [s]')
    ax = plt.gca()
    ax.xaxis.set_major_locator(fmt_month)
    plt.grid(True)
    plt.tight_layout()

    save_plot(os.path.join(OUTPUTDIR,'synthetic_delays.png'),
              h=ax, size=(8,6) )

