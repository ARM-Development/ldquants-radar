#!/usr/bin/env python
"""
 NAME:
   ldquantsRADAR.py

 PURPOSE:
   To read in the NWS Houston (KHGX) NEXRAD reflectivity archived data 
     along with the LDQUANTs laser disdrometer data for comparison. 

 SYNTAX:
   python ldquantsRADAR.py KHGXYYYYMMDD_HHMMSS_V06.ar2v houldquantsM1.c1.YYYYMMDD.HHMMSS.nc houldquantsS1.c1.YYYYMMDD.HHMMSS.nc

 INPUT::
   KHGXYYYYMMDD_HHMMSS_V06.ar2v        - NEXRAD Level 2 archive file.
   houldquantsM1.c1.YYYYMMDD.HHMMSS.nc - LDQUANTS (M1 Site) netCDF file. 
   houldquantsS1.c1.YYYYMMDD.HHMMSS.nc - LDQUANTS (S1 Site) netCDF file. 

 KEYWORDS:

 EXECUTION EXAMPLE:
   Linux example: python ldquantsRADAR.py KHGX20220109_150120_V06.ar2v houldquantsM1.c1.20220109.000000.nc houldquantsS1.c1.20220109.000000.nc

 MODIFICATION HISTORY:
   2022/02/18 - Joe O'Brien <obrienj@anl.gov> :
              Created using examples from the Py-Art quick start guide and LDQUANTS documentation

 NOTES:
   1) PyART quick start guide:
       https://arm-doe.github.io/pyart/
   2) NOAA NEXRAD ARCHIVE:
       ncdc.noaa.gov/nexradinv/
   3) PyART Github 
       github.com/ARM-DOE/pyart
   4) LDQUANTS documentation
       - https://www.arm.gov/capabilities/vaps/ldquants
   5) Github used to store this analysis
       - https://github.com/ARM-Development/ldquants-radar
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import pyart

from netCDF4 import Dataset

#-------------------------
# I) Define Functions 
#-------------------------

# Command line syntax statement. 
def help_message():
        """
        NAME: help_message

        PURPOSE: 
          To display to the command line the expected syntax for this program, including any/all keywords or options
        INPUT:
          N/A
        OUTPUT:
          Print statements to the command line and closing of the program.
        """ 
        print('\n')
        print('Syntax: ldquantsRADAR.py <-h> NEXRAD_Level2_file LDQUANTS_M1_file LDQUANTS_S1_file\n')
        print('      INPUT:  ')
        print('          NEXRAD_Level2_file:  - NEXRAD Level 2 archive file \n')
        print('      OPTIONS:  ')
        print('         -h:         - Help statement. Print Syntax\n')
        print('       KEYWORDS: \n')
        print('    EXAMPLE: python ldquantsRADAR.py KHGX20220109_150120_V06.ar2v houldquantsM1.c1.20220109.000000.nc \
        houldquantsS1.c1.20220109.000000.nc\n')

# Distance between radar and target - (distance between two points on a sphere).  
def sphereDistance(radLat,tarLat,radLon,tarLon):
	"""
	NAME: sphereDistance 
	
	PURPOSE: Calculation of the great circle distance between radar and target
	
	ASSUMPTIONS: 
		- Radius of the Earth = 6371 km / 6371000 meters
		- Distance is calculated for a smooth sphere 
		- Radar and Target are at the same altitude (need to check)

	INPUT: 
		i)   radLat - latitude of the radar in degrees [float or int]
		ii)  tarLat - latidude of the target in degrees [float or int]
		iii) radLon - longitude of the radar in degrees [float or int]
		iv)  tarLon - longitude of the target in degress [float or int]
	
	OUTPUT: 
		distance - distance between radar and target  in meters
	"""
	# convert latitude/longitudes to radians 
	radLat = radLat * (np.pi/180.)		# [radians]
	tarLat = tarLat * (np.pi/180.)		# [radians]
	radLon = radLon * (np.pi/180.)		# [radians]
	tarLon = tarLon * (np.pi/180.)		# [radians]

	# difference in latitude  - convert from degrees to radians 
	dLat = (tarLat-radLat)                  # [radians]
	# difference in longitude - convert from degress to radians
	dLon = (tarLon-radLon)                  # [radians]

	# Haversince formula
	numerator = (np.sin(dLat/2.0)**2.0)+np.cos(radLat)*np.cos(tarLat)*(np.sin(dLon/2.0)**2.0)
	distance = 2*6371000*np.arcsin(np.sqrt(numerator)) 		# [meters]

	# return the output
	return distance 

# Great Circle Bearing Calculation - Forward Azimuth Angle
def forAzimuth(radLat, tarLat, radLon, tarLon):
	"""
	NAME: forAzimuth 
	
	PURPOSE: Calculation of inital bearing along a great-circle arc
		 Known as Forward Azimuth Angle
	
	ASSUMPTIONS: 
		- Radius of the Earth = 6371 km / 6371000 meters
		- Distance is calculated for a smooth sphere 
		- Radar and Target are at the same altitude (need to check)

	INPUT: 
		i)   radLat - latitude of the radar in degrees [float or int]
		ii)  tarLat - latidude of the target in degrees [float or int]
		iii) radLon - longitude of the radar in degrees [float or int]
		iv)  tarLon - longitude of the target in degress [float or int]
	
	OUTPUT: 
		azimuth - forward azimuth angle in degrees 
	"""
	
	# convert latitude/longitudes to radians 
	radLat = radLat * (np.pi/180.)		# [radians]
	tarLat = tarLat * (np.pi/180.)		# [radians]
	radLon = radLon * (np.pi/180.)		# [radians]
	tarLon = tarLon * (np.pi/180.)		# [radians]
	
	# Differnce in longitudes 
	dLon = tarLon-radLon			# [radians]
		
	# Determine x,y coordinates for arc tangent function 
	y = np.sin(dLon)*np.cos(tarLat)
	x = (np.cos(radLat)*np.sin(tarLat))-(np.sin(radLat)*np.cos(tarLat)*np.cos(dLon))
	
	# Determine forward azimuth angle 
	azimuth = np.arctan2(y,x)               # [radians]

	# Return the output 
	return (azimuth*(180./np.pi))		# [degrees]
 
#-------------
# II) Input
#-------------

# Define the starting time of the code.
t0 = time.time()

# Check for options within the system arguments
for param in sys.argv:
  if param.startswith('-h'):
    help_message()
    sys.exit()
             
# Check to make sure there were three input files.
if (len(sys.argv) < 4):
    help_message()
    exit()
else:
  radfile = sys.argv[-3]
  m1file  = sys.argv[-2]
  s1file  = sys.argv[-1]
  #print("NEXRAD RADAR FILE:  ", radfile)
  #print("LDQUANTS M1 FILE:   ", m1file)
  #print("LDQUAHTS S1 FILE:   ", s1file)

#----------------------------
# III) Read the Radar Data
#----------------------------

# open the file, create the displays and figure
radar = pyart.io.read_nexrad_archive(radfile)

# Create the Radar Display
display = pyart.graph.RadarDisplay(radar)

# Calculate estimated rainfall rate from reflectivty
rain = pyart.retrieve.qpe.est_rain_rate_z(radar)

# Add the estimated rainfall rate back into the radar object
radar.add_field('est_rainfall_rate',rain)

#---------------------------------------
# IV) Read the LDQUANTS Data - M1 site
#---------------------------------------

# Read the file. Define a dataset variable.
nc_M = Dataset(m1file,'r')

# Define the NetCDF variables.
nc_vars = [var for var in nc_M.variables]

#-------------------------------------------------------------------------------------------
# V) Calculate distance and azimuth angle from radar to target. Retrieve Column above Target 
#-------------------------------------------------------------------------------------------

# call the sphereDistance function 
distance = sphereDistance(radar.latitude['data'][:],nc_M.variables['lat'][:],radar.longitude['data'],nc_M.variables['lon'][:])

# call the forAzimuth function 
azimuth  = forAzimuth(radar.latitude['data'][:],nc_M.variables['lat'][:],radar.longitude['data'],nc_M.variables['lon'][:])

# Find the reflectivty data for the azimuth over the target, find gates for that location
# Note: x,y,z gates are in km 
rhiData,rhi_x,rhi_y,rhi_z = pyart.graph.RadarDisplay(radar)._get_azimuth_rhi_data_x_y_z('reflectivity',azimuth,\
                                        edges=True, mask_tuple=None,gatefilter=None,filter_transitions=True)

# Find the estimated rainfall data for the azimuth over the target, find gates for that location
# Note: x,y,z gates are in km 
rainData,rain_x,rain_y,rain_z = pyart.graph.RadarDisplay(radar)._get_azimuth_rhi_data_x_y_z('est_rainfall_rate',azimuth,\
                                        edges=True, mask_tuple=None,gatefilter=None,filter_transitions=True)

# Calculate distance from the x,y coordinates to target
rhiDis  = np.sqrt((rhi_x**2)+(rhi_y**2))*np.sign(rhi_y)

# calculate the gate distance to the target, mask all data in RHI besides vertical column above the target
# need to iterate over every sweep
column_rain = np.ma.zeros(rhiData.shape[0])
for i in range(rhiData.shape[0]):
	
	# Find closest gate to the target
	tarGate = np.where(abs(rhiDis[i,:]-(distance/1000.)) == min(abs(rhiDis[i,:]-(distance/1000.))))
	# Create a new bool array to hold the new mask
	mask = np.ones(rhiData[i,:].shape,bool)
	mask[:] = True
	mask[tarGate[0]] = False
	# Append each gates estimated rain rate to the zeros array. 
	if rainData[i,tarGate[0]].mask == False:
		column_rain[i] = rainData[i,tarGate[0]]
	else:
		column_rain[i] = np.ma.masked
	# Mask every value but the gate over the target
	rhiData[i,:].mask  = mask
	rainData[i,:].mask = mask

#-------------------------------------------------------------------
# VI) Time Sync the Radar object and LDQUANTS file
#-------------------------------------------------------------------

# Note: for now, only radar input is a single file.
#       Time sync against daily file of the ldquants data.
#	Time sync is going to take more thought out approach.
#       Just trying to get a quick comparison with the disdrometer 

# I'm sure there's a better way to do this with netCDF4 datetime objects
# Convert radar time to seconds from midnight
# note: radar time units are seconds from start of the scan. 
beginTime     = radar.time['units'].split(' ')[2].split('T')[-1][0:-1]
sfm_beginTime = int(beginTime.split(':')[0])*3600.+int(beginTime.split(':')[1])*60.+int(beginTime.split(':')[2])
sfm_endTime   = sfm_beginTime+radar.time['data'][-1]

# find the ldqaunt times that match the radar sfm. 
# note: ldquant times are already in sfm from the start of the day
sync_x = np.where(nc_M.variables['time'][:] > sfm_beginTime)
sync_y = np.where(nc_M.variables['time'][:] < sfm_endTime)
time_sync = np.intersect1d(sync_x,sync_y)

#-----------------
# VII) Plot
#-----------------

# create the figure
fig, ax = plt.subplots(1,2,figsize=(12.80,6.45))

#------
# PPI 
#------

# Plot a PPI of the lowest sweep. 
# For whatever reason, pyart.graph.RadarMapDisplay.plot_ppi_map was not playing well with subplots
#sweep = 0 
display.plot_ppi('reflectivity',sweep=0,ax=ax[0])
# Determine limits for the PPI plot, force to be square
ax[0].set_ylim(-100,100)
ax[0].set_xlim(-100,100)
# Define a point on the graph for the LDQUANTS site. 
display.plot_label('M1',(nc_M.variables['lat'][:],nc_M.variables['lon'][:]), ax=ax[0], symbol='ro', text_color='k')
# Define a range ring around the PPI plot to coincide with psendo-RHI limits
##display.plot_range_ring(50,npts=100,ax=ax[0])

#------------
# psendo-RHI
#------------

# Plot the psendo-RHI using the forward azimuth angle calculated from forAzimuth function.
display.plot_azimuth_to_rhi('reflectivity',azimuth, ax=ax[1])
# Since this is disdrometer-radar comparison, shortening up plot limits
ax[1].set_xlim(0,50)
ax[1].set_ylim(0,20)
# Highlight the locaiton of the disdrometer data using the distance calculated 
#	in the sphereDistance function
ax[1].axvline((distance/1000.),0,1, color='k')    # Note: plot_azimuth_to_rhi x-axis is in km
# Plot text of the estimated rainfall rate from the radar gates over the disdrometer site. 
ax[1].text((distance/1000.)+1,19,'Radar RR: '+str(np.around(np.sum(column_rain),2))+' [mm/hr]',fontsize=9)
ax[1].text((distance/1000.)+1,18,'     M1 RR: '+str(np.around(np.sum(nc_M.variables['rain_rate'][time_sync]),2))+' [mm/hr]',\
	fontsize=9)

# Save the figure.
nout = 'ldquantsRadar_'+radar.metadata['instrument_name']+'_'+\
	radar.time['units'].split(' ')[-1].split('T')[0]+'_'+beginTime[0:2]+beginTime[3:5]+beginTime[6:8]+'.png' 
#plt.savefig(nout, dpi=100, bbox_inches='tight')
plt.savefig(nout, dpi=100)

#--------------------
# IV) END OF PROGRAM 
#--------------------

# Close the netCDF file. 
nc_M.close()

# define the ending time of the program for testing purposes 
t1 = time.time()

# print out run time
x = (t1-t0)/60.
print("Run Time: ", x, ' min')

