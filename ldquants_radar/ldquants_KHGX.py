#!/usr/bin/env python
"""
 NAME:
   ldquants_KHGX.py

 PURPOSE:
   To read in the NWS Houston (KHGX) NEXRAD reflectivity archived data
   along with the LDQUANTs laser disdrometer data for comparison.

 SYNTAX:
   python ldquants_KHGX.py LDQUANTS_Dir KHGX_Dir

 INPUT:
   LDQUANTS_Dir - directory containing the LDQUANTS (VDISQUANTS) data
                  from the S1 and M1 TRACER sites.
   KHGX_Dir - directory containing the KHGX Level 2 NEXRAD data

 KEYWORDS:

 EXECUTION EXAMPLE:
   Linux example: python ldquants_KHGX.py
                  /ARM/VAPS/LDQUANTS/data
                  /radar/KHGX/alldata

 MODIFICATION HISTORY:
   2022/04/26 - Joe O'Brien <obrienj@anl.gov> :
                Created using examples from Py-ART

 NOTES:
   1) PyART quick start guide:
       https://arm-doe.github.io/pyart/
   3) PyART Github
       github.com/ARM-DOE/pyart
   4) LDQUANTS documentation
       - https://www.arm.gov/capabilities/vaps/ldquants
   5) Github used to store this analysis
       - https://github.com/ARM-Development/ldquants-radar
"""

import sys
import time
import os
import datetime
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import netCDF4
import pyart
import xarray as xr
from pyart.util import columnsect

# -------------------------
# I) Define Functions
# -------------------------


def help_message():

    """
    To display to the command line the expected syntax for this
    program including any/all keywords or options

    Returns
    -------
    Print statements to the command line and closing of the program.
    """
    print("\n")
    print("Syntax: ldquants_KHGX.py <-h> <-p> LDQUANTS_DIR"
          + " KHGX_DIR\n")
    print("  INPUT:  ")
    print("    LDQUANTS_DIR: "
          + " - Directory to LDQUANTS data")
    print("    KHGX_DIR: "
          + " - Directory to KHGX data\n")
    print("  OPTIONS:  ")
    print("    -h:"
          + " - Help statement. Print Syntax")
    print("    -p:"
          + " - Plot PPI_RHI for each radar scan\n")
    print("  KEYWORDS: \n")
    print("EXAMPLE: python ldquants_KHGX.py"
          + " /ARM/VAP/LDQUANTS/alldata/"
          + " /radar/KHGX/alldata/\n")


def rain_rate_from_z_ds(ds, alpha=0.0376, beta=0.6112,
                        refl_field="reflectivity", rr_field="rain_rate_z"):
    """
    As the Py-ART QPE retrievals require a radar object, recreation of these
    functions are desirable for just a radar column subset.

    Various QPE relationship to calculate rain rate given an xarray dataset.

    Parameters:
    -----------
    ds : Xarray Dataset
        Dataset containing the extracted radar column above a given location.
    alpha, beta : floats, optional
        Factor (alpha) and exponent (beta) of the power law.
    refl_field : str, optional
        Name of the reflectivity field to use.
    rr_field : str, optional
        Name of the rainfall rate field.

    Returns:
    --------
    ds : Xarray Dataset
        Returns the inputed dataset array with a new rain rate dataarray
    """
    rain_z = ds[refl_field].copy()

    rr_data = alpha*np.ma.power(np.ma.power(10., 0.1 * rain_z), beta)

    ds[rr_field] = (['epoch', 'height'], rr_data)

    field_attrs = {"units" : "mm/h",
                   "standard_name" : "rain_rate_z",
                   "long_name" : "rainfall_rate_from_z",
                   "valid_min" : 0,
                   "valid_max" : 500,
                   "alpha" : alpha,
                   "beta" : beta}

    ds[rr_field].attrs = field_attrs

    return ds

def plot_ppirhi(display, nradar, ldquants, tsync, nazimuth,
                ndistance, ncolumn):

    """
    To display the PPI and peusdo-RHI scan for a given PyART
    RadarDisplay object for comparison with a ground site
    or target.

    Parameters
    ----------
    display   : RadarDisplay
                PyART RadarDisplay object from which reflectivity factor
                is displayed
    nradar    : Radar
                PyART Radar object form which metadata/time are used for
                filename creation
    ldquants  : netCDF4
                netCDF object of the VAP LDQUANTs data from which
                lat/lon and rain rate are displayed.
    tsync     : masked array
                numpy masked array of indices where time between
                ldquants/radar match
    nazimuth  : float, [degrees]
                azimuth value to display with peusdo-RHI scan
    ndistance : float, [km]
                distance from the radar to observed column
    ncolumn   : array
                radar derived rain rates for column above observation site

    Returns
    -------
        plot : *png
               Saves *png file for the plot.
    """
    # create the figure
    fig, axarr = plt.subplots(1, 2, figsize=(12.80, 6.45))
    # Plot a PPI of the lowest sweep.
    # For whatever reason, pyart.graph.RadarMapDisplay.plot_ppi_map
    # was not playing well with subplots
    display.plot_ppi('reflectivity', sweep=0, axarr=axarr[0])
    # Determine limits for the PPI plot, force to be square
    axarr[0].set_ylim(-100, 100)
    axarr[0].set_xlim(-100, 100)
    # Define a point on the graph for the LDQUANTS site.
    display.plot_label('M1', ldquants.variables['lat'][:],
                       ldquants.variables['lon'][:], ax=axarr[0], symbol='ro',
                       text_color='k')
    # Plot the psendo-RHI using the forward azimuth angle calculated
    # from for_azimuth function.
    display.plot_azimuth_to_rhi('reflectivity', nazimuth, ax=axarr[1])
    # Since this is disdrometer-radar comparison, shortening up plot limits
    axarr[1].set_xlim(0, 50)
    axarr[1].set_ylim(0, 20)
    # Highlight the locaiton of the disdrometer data using the distance
    # calculated in the sphere_distance function
    axarr[1].axvline((ndistance/1000.), 0, 1, color='k')
    # Plot text of the estimated rainfall rate from the radar gates over
    # the disdrometer site.
    axarr[1].text((ndistance/1000.) + 1, 19, 'Radar RR: '
                  + str(np.around(np.sum(ncolumn), 2))
                  + ' [mm/hr]', fontsize=9)
    axarr[1].text((ndistance/1000.) + 1, 18, '     M1 RR: '
                  + str(np.around(np.sum(ldquants.variables['rain_rate']
                                         [tsync]), 2))
                  + ' [mm/hr]', fontsize=9)
    # Save the figure.
    nout = ('ldquantsRadar_' + nradar.metadata['instrument_name'] + '_'
            + nradar.time['units'].split(' ')[-1].split('T')[0] + '_'
            + begin_time[0:2] + begin_time[3:5] + begin_time[6:8]+'.png')
    plt.savefig(nout, dpi=100)
    # Close the figure.
    plt.close(fig)


def plot_hist2d(ds_radar, ds_ld, brange, nbin):

    """
    To calculate the bi-dimensional histogram of two data samples
    and display

    Parameters
    ----------
    ds_radar : xarray Dataset
             Dataset containing the radar fields of the lowest gate above
             the disdrometer site.
    ds_ld : xarray Dataset
             Dataset containing the disdrometer data that corresponds to the
             radar fields
    brange : list
             List in the format [xmin, xmax, ymin, ymax] for values
             to be binned within a 2D histogram
    nbin   : int,float
             value for the number of bins to apply to the 2D histogram

    Returns
    -------
    plot   : *png
             Saves *png file for the plot.
    """
    # create the subplots
    fig, axarr = plt.subplots(2, 1, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.2)
    # create short name for reflectivity
    reflect = 'reflectivity_factor_sband20c'
    # Plot the data.
    counts, xedges, yedges, im = axarr[0].hist2d(ds_ld[reflect][:],
                                                 ds_radar.reflectivity[:],
                                                 range=[[brange[0],
                                                         brange[1]],
                                                        [brange[2],
                                                         brange[3]]],
                                                 bins=nbin,
                                                 norm=mpl.colors.LogNorm(),
                                                 cmap=mpl.cm.jet)
    # Add a 1:1 ratio line
    axarr[0].plot(np.arange(brange[0] - 10, brange[1] + 10, 1),
                  np.arange(brange[0] - 10, brange[1] + 10, 1), 'k')
    # Define axe titles for the figure.
    axarr[0].set_ylabel(r'KHGX $Z_e$ [dBZ]')
    axarr[0].set_xlabel(r'LDQUANTS Derived $Z_e$ [dBZ]')
    # Display the colorbar and colorbar label
    fig.colorbar(im, ax=axarr[0], label='Counts [#]')

    # Plot the data.
    # Create short name for rain rates
    rain = 'rain_rate_z'
    print('rain_rate max: ', np.max(ds_ld.rain_rate), np.max(ds_radar.rain_rate_z))
    counts, xedges, yedges, im = axarr[1].hist2d(ds_ld.rain_rate[:],
                                                 ds_radar[rain][:],
                                                 range=[[0, 5], [0, 5]],
                                                 bins=nbin,
                                                 norm=mpl.colors.LogNorm(),
                                                 cmap=mpl.cm.jet)
    # Add a 1:1 ratio line
    axarr[1].plot(np.arange(brange[0] - 10, brange[1] + 10, 1),
                  np.arange(brange[0] - 10, brange[1] + 10, 1), 'k')
    axarr[1].set_xlabel(r'LDQUANTS Rain Rate [mm/hr]')
    axarr[1].set_ylabel(r'KHGX Derived Rain Rate [mm/hr]')
    # Plot the colorbar and colorbar label
    fig.colorbar(im, ax=axarr[1], label='Counts [#]')

    # plot the title
    date_zero = datetime.datetime.utcfromtimestamp(
                                  ds_radar['epoch'].data[0]).strftime('%Y%m%d')
    date_last = datetime.datetime.utcfromtimestamp(
                                  ds_radar['epoch'].data[-1]).strftime('%Y%m%d')
    if date_zero != date_last:
        plt.suptitle('KHGX-LDQUANTS Comparison (' + date_zero + ' - '
                     + date_last + ')')
        nout = ('ldquants_KHGX_' + date_zero + '_' + date_last
                + '_fullComparison.png')
        plt.savefig(nout, dpi=200)
        plt.close(fig)
    else:
        plt.suptitle('KHGX-LDQUANTS Comparison (' + date_zero + ' )')
        # Save the figure.
        nout = ('ldquants_KHGX' + '_' + date_zero + '_fullComparison.png')
        plt.savefig(nout, dpi=200)
        # Close the figure.
        plt.close(fig)


def plot_timeseries(column, ld):
    """
    To display the reflectivity above the disdrometer site
    along with associated laser disdrometer information.
    """
    fig, axs = plt.subplots(1, 2, figsize=[12,6], sharex=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    #-----------------------------------------------
    # Reflectivity over the disdrometer site
    #-----------------------------------------------
    column.reflectivity.plot(x='epoch', ax=axs[0],
                             cmap='pyart_HomeyerRainbow',
                             vmin=-20,
                             vmax=40)
    #col_start = datetime.datetime.fromtimestamp(column.epoch[0])
    #col_end = datetime.datetime.fromtimestamp(column.epoch[-1])
    axs[0].set_xlim([column.epoch[0], column.epoch[-1]])
    # Set the labels
    axs[0].set_xlabel('Date [HHMM UTC MM/DD/YYYY]')
    axs[0].set_ylabel('Height Above Disdrometer [meters]')
    # Grab the xticks
    xticks = axs[0].get_xticks()
    # Define new ticks
    newticks = []
    for tick in xticks:
        newticks.append(datetime.datetime.
                        utcfromtimestamp(tick).strftime("%H%M UTC %m/%d/%Y"))
    axs[0].set_xticklabels(newticks)

    #------------------------------------------------
    # Disdrometer Rain rate and Radar rain rate
    #------------------------------------------------
    ld.rain_rate.plot(x='epoch')
    axs[1].set_xlim([ld.epoch[0], ld.epoch[-1]])
    # Set the labels
    axs[1].set_xlabel('Date [HHMM UTC MM/DD/YYYY]')
    axs[1].set_ylabel('Rain Rate [mm/hr]')
    # Grab the xticks
    xticks = axs[1].get_xticks()
    # Define new ticks
    newticks = []
    for tick in xticks:
        newticks.append(datetime.datetime.
                        utcfromtimestamp(tick).strftime("%H%M UTC %m/%d/%Y"))
    axs[1].set_xticklabels(newticks)
    # Set the axe labels
    axs[1].set_xlabel('Date [HHMM UTC MM/DD/YYYY]')
    #axs[1].set_ylabel('Height Above Disdrometer [meters]')
    # format for dates
    fig.autofmt_xdate()
    # Twin Y-Axis for Accumulation
    ax2 = axs[1].twinx()
    p1, = ax2.plot(ld.epoch, ld['rain_rate'].cumsum()*(1/60.), color='orange')
    ax2.set_ylabel("Accumulation [mm]")
    ax2.yaxis.label.set_color(p1.get_color())

    # plot the title
    date_zero = datetime.datetime.utcfromtimestamp(
                                  column['epoch'].data[0]).strftime('%Y%m%d')
    date_last = datetime.datetime.utcfromtimestamp(
                                  column['epoch'].data[-1]).strftime('%Y%m%d')
    if date_zero != date_last:
        plt.suptitle('KHGX-LDQUANTS Timeseries ('+ date_zero + ' - '
                     + date_last + ')')
        nout = ('ldquants_KHGX_timeseries_' + date_zero + '_' + date_last
                + '.png')
        plt.savefig(nout, dpi=200)
        plt.close(fig)
    else:
        plt.suptitle('KHGX-LDQUANTS Timeseries (' + date_zero + ')')
        nout = ('ldquants_KHGX_timeseries_' + date_zero + '.png')
        plt.savefig(nout, dpi=200)
        plt.close(fig)


# -------------------------
# II) Input
# -------------------------

# Define the starting time of the code.
t0 = time.time()

# Plot Flag
pflag = False

# Check for options within the system arguments
for param in sys.argv:
    if param.startswith('-h'):
        help_message()
        sys.exit()
    if param.startswith('-p'):
        pflag = True

# Check to make sure there were three input files.
if len(sys.argv) < 1:
    help_message()
    sys.exit()
else:
    ld_dir = sys.argv[-2]
    rad_dir = sys.argv[-1]

# Define the present working directory.
pwd = os.getcwd()

# -------------------------------------------------------------------
# IV) Iterate over successful downloads.
#     Calculate distance and azimuth to target
# -------------------------------------------------------------------

# initiate a dictionary to hold the striped out data
# from each scan.
ld_data = {'epoch': [], 'rain_rate': [], 'reflectivity_factor_sband20c': [],
           'lwc': [], 'total_droplet_concentration': [], 'med_diameter': [],
          }

# Change directory to the LDQUANTS file directory.
os.chdir(ld_dir)
# blank list to hold all the xrray datasets before merging
ds_all = []
ds_one = []
# iterate over the local files.
for file in sorted(glob.glob("*.nc")):
    # check to make sure which file we are using.
    ld_date = file.split('.')[2]
    # Grab the lat/lon from the LDQUANTS file.
    nc_fid = netCDF4.Dataset(file)
    # Find the times for the file (epoch)
    nc_times = nc_fid['time_offset'][:].data + nc_fid['base_time'][:].data
    # change to radar directory
    os.chdir(pwd+'/'+rad_dir)
    # check to see which radar file matches
    glob_tmp = "KHGX"+ld_date+"*"
    for rad in sorted(glob.glob(glob_tmp)):
        if rad[-3:] != "MDM":
            print('YES-Match: ', rad, file)
            # read the radar file into PY-ART object
            radar = pyart.io.read(rad)
            # Grab the radar column above the LDQUANTS
            ld_lat = float(nc_fid.variables['lat'][:].data)
            ld_lon = float(nc_fid.variables['lon'][:].data)
            column = columnsect.get_field_location(radar, ld_lat, ld_lon)
            # Add a datetime dimension to the xrray column.
            dt = datetime.datetime.strptime(column.date+' '+column.time[:-1],
                                            "%Y-%m-%d %H:%M:%S")
            coor_ad = column.assign_coords(epoch=dt.timestamp())
            ncolumn = coor_ad.expand_dims('epoch')
            # Make sure there are no duplicate height indices
            index = np.unique(column['height'], return_index=True)
            ncolumn = ncolumn.isel(height=index[1])
             # Interpolate heights
            ncolumn.interp(height=np.arange(100, 3000, 100))
            # Append to the xrray dataset list. Interpolate heights
            ds_all.append(ncolumn.interp(height=np.arange(100, 3000, 100)))
            # Determine where the LDQUANTS data matches the epoch time
            syncx = np.where(nc_times >= dt.timestamp())
            syncy = np.where(nc_times <=
                            (dt.timestamp() + radar.time['data'][-1]))
            sync = np.intersect1d(syncx, syncy)
            # Append to the LDQUANTS data directory
            ld_data['epoch'].append(dt.timestamp())
            for key in ld_data:
                if key != 'epoch':
                    ld_data[key].append(np.ma.average(
                                        nc_fid.variables[key][sync]))
    # change back to LDQUANTS file directory
    os.chdir(pwd+'/'+ld_dir)
    # Close the netcdf file.
    nc_fid.close()

# Combine the dataset to merge all columnar data.
ds = xr.concat(ds_all, 'epoch')

# Convert the LD dictionary to xarray dataset
container = []
for key in ld_data:
    if key != 'epoch':
        da = xr.DataArray(ld_data[key], coords=[ld_data['epoch']],
                          name=key, dims=['epoch'])
        container.append(da)
ld_ds = xr.merge(container)

# Calculate rain rate for the extracted column
dsA = rain_rate_from_z_ds(ds)

# -----------------------------------------------
# V) Plot the 2D-Histrogram for the entire date
# ----------------------------------------------

# Define the range of values to be binned
his_range = [-30, 70, -30, 70]
NBINS = 100

# plot the timeseries
plot_timeseries(ds, ld_ds)

# plot the histogram
plot_hist2d(ds.isel(height=1), ld_ds, his_range, NBINS)

# --------------------
# VI) END OF PROGRAM
# --------------------

# define the ending time of the program for testing purposes
t1 = time.time()

# print out run time
t3 = (t1-t0)/60.
print("Run Time: ", t3, ' min')
