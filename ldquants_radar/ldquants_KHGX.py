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
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.feature import NaturalEarthFeature

import netCDF4
import pyart
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


def plot_his(data_m1, data_s1, brange, nbin):

    """
    To calculate the bi-dimensional histogram of two data samples
    and display

    Parameters
    ----------
    data_m1 : dict
             Dictionary containing int/float values of radar field
             parameters from the Houston NEXRAD radar and M1-site
             LDQUANTS data that are desired to be displayed
    data_s1 : dict
             Dictionary containing int/float values of radar
             field parameters from the Houston NEXRAD radar and
             M1-site LDQUANTS data that are desired to be displayed
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
    fig, axarr = plt.subplots(2, 2, figsize=(12.80, 6.45))
    # Plot the data.
    axarr[0, 0].hist2d(data_m1['LD_Z'], data_m1['rhi_data'],
                       range=[[brange[0], brange[1]], [brange[2], brange[3]]],
                       norm=mpl.colors.LogNorm(),
                       cmap=mpl.cm.jet)
    # Add a 1:1 ratio line
    axarr[0, 0].plot(np.arange(brange[0] - 10, brange[1] + 10, 1),
                     np.arange(brange[0] - 10, brange[1] + 10, 1), 'k')
    # Define axe titles for the figure.
    axarr[0, 0].set_ylabel(r'KHGX $Z_e$ [dBZ]')
    axarr[0, 0].set_xlabel(r'LDQUANTS M1-Site Derived $Z_e$ [dBZ]')
    # Plot the data.
    axarr[0, 1].hist2d(data_m1['VD_Z'], data_m1['rhi_data'],
                       range=[[brange[0], brange[1]], [brange[2], brange[3]]],
                       norm=mpl.colors.LogNorm(), cmap=mpl.cm.jet)
    axarr[0, 1].plot(np.arange(brange[0] - 10, brange[1] + 10, 1),
                     np.arange(brange[0] - 10, brange[1] + 10, 1), 'k')
    axarr[0, 1].set_xlabel(r'VDISQUANTS M1-Site Derived $Z_e$ [dBZ]')
    axarr[0, 1].set_ylabel(r'KHGX $Z_e$ [dBZ]')
    # Plot the data.
    axarr[1, 0].hist2d(data_s1['LD_Z'], data_s1['rhi_data'],
                       range=[[brange[0], brange[1]], [brange[2], brange[3]]],
                       norm=mpl.colors.LogNorm(), cmap=mpl.cm.jet)
    axarr[1, 0].plot(np.arange(brange[0] - 10, brange[1] + 10, 1),
                     np.arange(brange[0] - 10, brange[1] + 10, 1), 'k')
    axarr[1, 0].set_xlabel(r'LDQUANTS S1-Site Derived $Z_e$ [dBZ]')
    axarr[1, 0].set_ylabel(r'KHGX $Z_e$ [dBZ]')
    # Hide the blank subplot
    axarr[1, 1].set_visible(False)
    # plot the title
    if data_m1['date'][0] == data_m1['date'][-1]:
        plt.suptitle(data_m1['date'][0]
                     + ' Equivalent Radar Reflectivity Factor Comparison')
    else:
        plt.suptitle(data_m1['date'][0] + ' - ' + data_m1['date'][-1]
                     + ' Equivalent Radar Reflectivity Factor Comparison')
    # Save the figure.
    nout = ('ldquantsRadar_' + radar.metadata['instrument_name'] + '_'
            + radar.time['units'].split(' ')[-1].split('T')[0]
            + 'fullComparison.png')
    plt.savefig(nout, dpi=100)
    # Close the figure.
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
print(ld_dir)
print(rad_dir)

# -------------------------------------------------------------------
# IV) Iterate over successful downloads.
#     Calculate distance and azimuth to target
# -------------------------------------------------------------------

# initiate a dictionary to hold the striped out data
# from each scan.
ndata_m1 = {'date': [], 'time': [], 'xGate': [], 'yGate': [], 'zGate': [],
            'tarDis': [], 'tarAzi': [], 'rhi_data': [], 'rain_data': [],
            'LD_rain': [], 'LD_Z': [], 'VD_rain': [], 'VD_Z': [],
            }

ndata_s1 = {'date': [], 'time': [], 'xGate': [], 'yGate': [], 'zGate': [],
            'tarDis': [], 'tarAzi': [], 'rhi_data': [], 'rain_data': [],
            'LD_rain': [], 'LD_Z': []
            }

# Change directory to the LDQUANTS file directory.
os.chdir(ld_dir)
# blank list to hold all the xrray datasets before merging
ds_all = []
time1 = time.time()
# iterate over the local files.
for file in sorted(glob.glob("*.nc")):
    # check to make sure which file we are using. 
    ld_date = file.split('.')[2]
    # Grab the lat/lon from the LDQUANTS file. 
    nc_fid = netCDF4.Dataset(file)
    # change to radar directory
    os.chdir(pwd+'/'+rad_dir)
    # check to see which radar file matches
    for rad in sorted(glob.glob("KHGX*")):
        rad_date = rad.split("_")[0][4:]
        if rad_date == ld_date and rad.split("_")[-1] != "MDM":
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
            # Append to the xrray dataset list
            ds_all.append(ncolumn)
    # change back to LDQUANTS file directory
    os.chdir(pwd+'/'+ld_dir)
    # Close the netcdf file.
    nc_fid.close()

"""
for loc_file in downloads.iter_success():
    # check to make sure not to use the MDM files
    if loc_file.filepath[-3:] != 'MDM':
        # print the file
        print(loc_file)
        # Read in the file.
        radar = pyart.io.read(loc_file.filepath)
        # Create the Radar Display
        display = pyart.graph.RadarDisplay(radar)
        # Calculate estimated rainfall rate from reflectivty
        rain = pyart.retrieve.qpe.est_rain_rate_z(radar)
        # Add the estimated rainfall rate back into the radar object
        radar.add_field('est_rainfall_rate', rain)

        # Call the plotting function
        if pflag is True:
            plot_ppirhi(display, radar, nc_M, time_sync_M1,
                        azim_M1, dis_M1, colrain_m1)
"""
time2 = time.time()
print(time2-time1)
# -----------------------------------------------
# V) Plot the 2D-Histrogram for the entire date
# ----------------------------------------------

# Define the range of values to be binned
his_range = [-35, 70, -35, 70]
NBINS = 90

# call the plot_his function
#plot_his(ndata_m1, ndata_s1, his_range, NBINS)

# --------------------
# VI) END OF PROGRAM
# --------------------

# Close the netCDF file.
##nc_M.close()
##nc_S.close()
##nc_V.close()

# define the ending time of the program for testing purposes
t1 = time.time()

# print out run time
t3 = (t1-t0)/60.
print("Run Time: ", t3, ' min')
