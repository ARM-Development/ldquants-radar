#!/usr/bin/env python
"""
 NAME:
   find_ldquants.py

 PURPOSE:
   To read in all the LDQUANTs laser disdrometer to determine periods for analysis (rain rate > 50mm/hr). 

 SYNTAX:
   python find_ldquants.py 

 INPUT::

 KEYWORDS:

 EXECUTION EXAMPLE:
   Linux example: python find_ldquants.py 

 MODIFICATION HISTORY:
   2022/03/15 - Joe O'Brien <obrienj@anl.gov> :
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
   6) Notebook used for nexradaws data pull/display
       - https://github.com/scollis/notebooks/blob/master/urban/Boston%20Snow%20Retrieval.ipynb
   7) Python nexradaws documentation
       nexradaws.readthedocs.io/_/downloads/en/latest/pdf/
"""

import sys
import time
import numpy as np
import datetime 
import glob

from netCDF4 import Dataset
from os import path

#-------------------------
# I) Define Functions 
#-------------------------

# Command line syntax statement. 
def help_message():
        """
          To display to the command line the expected syntax for this program,
             including any/all keywords or options
        
        Returns
        -------
             Print statements to the command line and closing of the program.
        """ 
        print('\n')
        print('Syntax: find_ldquants.py <-h> \n')
        print('      INPUT:  ')
        print('      OPTIONS:  ')
        print('          -h:         - Help statement. Print Syntax\n')
        print('       KEYWORDS: \n')
        print('EXAMPLE: python find_ldquants.py \n')

#-------------------------
# II) Input
#-------------------------

# Define the starting time of the code.
t0 = time.time()

# Check for options within the system arguments
for param in sys.argv:
  if param.startswith('-h'):
    help_message()
    sys.exit()
             
# Define all the images within the current working directory. 
filelist = []
for file in sorted(glob.glob("*.nc")):
        filelist.append(file)

#---------------------------------------------------------
# III) Open each LDQUANTS files within a directory
#      Check for times where RR > 50 mm/hr  
#---------------------------------------------------------

# Define a dictionary to hold the date/time, rain rate and 
# calculated reflectivity factor for periods of rain. 
outData = {'date': [], 'time': [], 'site': [], 'rainRate': [],\
           'Z_sband': []}
 
# Loop through all the files. Open and search for rain rate. 
for nfile in filelist:
    # print the filename out for reference
    print(nfile)	
    # Read the file. Define a dataset variable.
    nc_M = Dataset(nfile,'r')

    # Determine if there are good rain rate values within this file. 
    goodRain = np.where(nc_M.variables['rain_rate'][:].mask != True)
    print(goodRain) 
    # Loop through periods of disdrometer data and see if
    # the threshold is met
    print(goodRain[0].shape[0])
    # First check if there is any observation that meets our threshold. 
    if goodRain[0].shape[0] > 0:
        for i in range(goodRain[0].shape[0]):
            if nc_M.variables['rain_rate'][:][goodRain][i] > 50:
                # Append the data to the dictionary
                #----------------------------------
                # Calculate the date of the file. 
                outData['date'].append(nc_M.variables['time'].units.split(' ')[2])
                # Calculate the time for each observation that meets our threshold.
                # Display the time as a HH-MM-SS string within the output dictionary
                outData['time'].append(str(datetime.timedelta(seconds=nc_M.variables\
                    ['time'][:][goodRain][i])))
                # Determine which site this data is from
                outData['site'].append(nfile.split('.')[0][-2:])
                # Store the rain rate data. 
                outData['rainRate'].append(nc_M.variables['rain_rate'][:][goodRain][i])
                # Store the S-band reflectivity data. 
                outData['Z_sband'].append(nc_M.variables['reflectivity_factor_sband20c']\
                    [:][goodRain][i])

    # Close the netCDF file. 
    nc_M.close()

#-----------------------
# IV) Write out the data
#-----------------------

# declare an output file name
# Write data to file
nout_name = 'ldquants_validTimes_25mm.txt'
nout = open(nout_name,'w')
for i in range(len(outData['date'])):
    nout.write(outData['date'][i]+','+outData['time'][i]+','+outData['site'][i]+','+\
    str(outData['rainRate'][i])+','+str(outData['Z_sband'][i])+'\n')

# Close the file.
nout.close()

#--------------------
# VI) END OF PROGRAM 
#--------------------

# Close the netCDF file. 
# define the ending time of the program for testing purposes 
t1 = time.time()

# print out run time
x = (t1-t0)/60.
print("Run Time: ", x, ' min')

