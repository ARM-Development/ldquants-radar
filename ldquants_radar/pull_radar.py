#!/usr/bin/env python
"""
 NAME:
   pull_radar.py

 PURPOSE:
   To pull NWS NEXRAD radar data for a given site and date and display .

 SYNTAX:
   python pull_radar.py [site=site] [date=date] [time=time] [output=output] 

 INPUT::
   N/A       - non-required due to keyword defaults

 KEYWORDS:
   site      - Define what radar to display. Default is KLOT.
	       Options are any NWS radar indicator 

   date      - date to pull data from in YYYY_MM_DD format
	       Default is current date. 

   time      - time to start pulling data from in HHMMSS [UTC]
	       Default is the the start of the day (i.e. midnight) [UTC].
   
   outdir    - path to save the image file(s) within. 
	       Default is home directory of the user.
  OPTIONS:
    -h       - display the help message to the command line

    -a,--all - download all radar scans for a given date. 
	       If date keyword is not set, will grab all 
               scans for the current date until time of 
               program execution.     

 EXECUTION EXAMPLE:
   Linux example: python pull_radar.py location=KGFK

 MODIFICATION HISTORY:
   2022/02/28 - Joe O'Brien <obrienj@anl.gov> :
              Created using examples from Scott Collis's snowfall rate github notebook
	      Has example for pulling data from an AWS server. 
   2022/03/03 - Joe O'Brien <obrienj@anl.gov> :
              Adding get_avail_scans_in_range call to pull all scans within a date. 
              Added path to downloads so it does not store in current working directory

 NOTES:
   1) PyART quick start guide:
       https://arm-doe.github.io/pyart/
   2) NOAA NEXRAD ARCHIVE:
       ncdc.noaa.gov/nexradinv/
   3) PyART Github 
       github.com/ARM-DOE/pyart
   4) Scott Collis' Boston Snow Retrieval.ipynb
       https://github.com/scollis/notebooks/blob/master/urban/Boston%20Snow%20Retrieval.ipynb
   5) Python nexradaws documentation
       nexradaws.readthedocs.io/_/downloads/en/latest/pdf/
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import pyart
import datetime

import nexradaws 
import tempfile
import cartopy.feature as cfeature

from cartopy.feature import NaturalEarthFeature
from os import path

#------------------------------------------------------------------------------------------------------
# I) Define Functions 
#------------------------------------------------------------------------------------------------------

def help_message():
  print('\n')
  print('Syntax: pull_radar.py <-h> <-a,--all> [site=site] [date=date] [time=time] [outdir=outdir] \n')
  print('      INPUT:  ')
  print('          Not needed due to defaults \n')
  print('      OPTIONS:  ')
  print('         -h:         - Help statement. Print Syntax')
  print('         -a,-all:    - Download all scans for the date\n')
  print('       KEYWORDS: ')
  print('         site:       - define what radar to display. ')
  print('		      - Default is Chicago, KLOT\n')
  print('         date:       - Date to pull date from YYYY-MM-DD format.')
  print('                     - Default is the current date.\n')
  print('         time:       - time to pull data from in HHMMSS [UTC]\n')
  print('         outdir:     - path to directory to save image files within\n')

# #https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date/32237949
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

# Taken from Scott's notebook file
# More tests need to be included. 
def pull_radar(connection, site, ndatetime, outdir=None):
    # Create a temporary directory to hold the data. 
    tlocation   = tempfile.mkdtemp()
    #-----------------------------------------------------
    # Check to see if date is available this site. 
    print(ndatetime.year,ndatetime.month,ndatetime.day)
    print(connection.get_avail_years())
    if str(ndatetime.year) not in connection.get_avail_years():
       print('ERROR: Date Not Found for Radar Site\n')
       sys.exit(1)
    else:
       # Need two digit month for the datetime object. 
       if ('%02d' % ndatetime.month)  not in connection.get_avail_months(ndatetime.year):
          print('ERROR: Date Not Found For Radar Site\n')
          sys.exit(1)
       else:
          if ('%02d' % ndatetime.day)  not in connection.get_avail_days(ndatetime.year,ndatetime.month):
             print('ERROR: Date Not Found For Radar Site\n')
             sys.exit(1)
    # Check to see if data is available for the site. 
    siteCheck = connection.get_avail_radars(ndatetime.year, ndatetime.month, ndatetime.day)
    if site not in siteCheck:
       print('ERROR: Radar Site Not Found\n')
       help_message()
       sys.exit(1)
    #-----------------------------------------------------
    # Check to see if date is available this site.
    # Use AWS server to grab the available scans based on the datetime object and site. 
    these_scans = connection.get_avail_scans(ndatetime.year,ndatetime.month, ndatetime.day, site)
    # Grab the times of the individual scans
    # produces a datetime object for each file
    these_times = [scan.scan_time for scan in these_scans]
    # Target datetime argument 
    targ        = ndatetime
    
    #Need to clean
    these_good_scans = []
    these_good_times = []
    for i in range(len(these_scans)):
        if these_times[i] is not None:
            these_good_times.append(these_times[i])
            these_good_scans.append(these_scans[i])
    # Make sure we grabbed files
    ##print(len(these_good_scans), len(these_good_times))
    # Check for the file nearest in time to our datetime object 
    this_nearest_time = nearest(these_good_times, targ)
    # Index of the scan closest to our datetime object
    this_index = these_good_times.index(this_nearest_time)
    # Download the scan closest to our datetime object locally.
    # Make sure a time exists 
    if outdir != None: 
          #localfiles = conn.download(these_good_scans[this_index],site)
          localfiles = connection.download(these_good_scans[this_index],outdir,keep_aws_folders=True)
    else:
          localfiles = connection.download(these_good_scans[this_index],site,keep_aws_folders=True)
    return pyart.io.read(localfiles.success[0].filepath)

# download all valid scans for a given date
def pull_all(connection, site, ndatetime, outdir=None):
    # Create a temporary directory to hold the data. 
    tlocation   = tempfile.mkdtemp()
    #------------------------------------------------------
    # Check to see if date is available this site. 
    if str(ndatetime.year) not in connection.get_avail_years():
       print('ERROR: Date Not Found for Radar Site\n')
       sys.exit(1)
    else:
       if ('%02d' % ndatetime.month) not in connection.get_avail_months(ndatetime.year):
          print('ERROR: Date Not Found For Radar Site\n')
          sys.exit(1)
       else:
          if ('%02d' % ndatetime.day) not in connection.get_avail_days(ndatetime.year,ndatetime.month):
             print('ERROR: Date Not Found For Radar Site\n')
             sys.exit(1)
    # Check to see if data is available for the site. 
    siteCheck = connection.get_avail_radars(ndatetime.year, ndatetime.month, ndatetime.day)
    if site not in siteCheck:
       print('ERROR: Radar Site Not Found\n')
       help_message()
       sys.exit(1)
    #-------------------------------------------------------
    # grab all the scans for the date. 
    # Use AWS server to grab the available scans based on the datetime object and site. 
    these_scans = connection.get_avail_scans(ndate.year,ndate.month,ndate.day,site)
    # Grab the times of the individual scans
    # produces a datetime object for each file
    these_times = [scan.scan_time for scan in these_scans]
    # Need to clean
    these_good_scans = []
    these_good_times = []
    for i in range(len(these_scans)):
        if these_times[i] is not None:
           these_good_times.append(these_times[i])
           these_good_scans.append(these_scans[i])
   
     # download the data. search if output directory is defined. 
    if outdir != None: 
        downloads = connection.download(these_good_scans,outdir,keep_aws_folders=True)
    else:
        downloads = connection.download(these_good_scans,site,keep_aws_folders=True)

    # return the downloaded files		  
    return downloads

#-------------------------------------------------------------------------------------
# II) Define Options
#-------------------------------------------------------------------------------------

# Define the starting time of the code
t0 = time.time()

# Define a default for the all option
n_all = False

# Check for options within the system arguments
for param in sys.argv:
  if param.startswith('-h'):
    help_message() 
    sys.exit()
  if param.startswith('-a') | param.startswith('-all') | param.startswith('--all'):
    n_all = True

# Check for keywords
# Declare a default 
nsite  = 'KLOT'

# Note date/time needs to be timezone aware!
# If date is not set, assume they want today and closest time to this program running. 
ndate = datetime.datetime.now(datetime.timezone.utc)

# Define a default output directory. 
outdir = None 

# Check command line arguments to see if additional choices were set. 
for param in sys.argv:
        # Radar Identifier
        if param.startswith('site='):
          nsite    = param.split('=')[-1]
        # Time nearest to scan HHMMSS
        if param.startswith('time='):
          tempTime = param.split('=')[-1]
          ndate = ndate.replace(hour=int(tempTime[0:2]),minute=int(tempTime[2:4]),\
                  second=int(tempTime[4:6]))
        # Date of desired Scan. 
        if param.startswith('date='):
          tempDate = param.split('=')[-1]
          # Check to make sure its a valid date. 
          try: 
              ndate = ndate.replace(year=int(tempDate.split('-')[0]),month=int(tempDate.split('-')[1]),\
                      day=int(tempDate.split('-')[2]))
          except ValueError:
              print("Error: Input Day is outside range for given Month")
              print("       Please input correct date\n")
              sys.exit(1)
        if param.startswith('outdir='):
          outdir = path.expanduser('~')+param.split('=')[-1]
          
#------------------------------------------------------------------------------
# III) AMAZON AWS Data Pull and Display
#------------------------------------------------------------------------------

# Connect to the Cloud - nexradaws is pup installable from anaconda
conn = nexradaws.NexradAwsInterface()

# See if the outdir directory has been set. 
if outdir == None:
   # check the all image flag. If set, find all scans and clean. 
   if n_all is True:
      downloads = pull_all(conn,nsite,ndate)
   else:
      # Read the file. 
      radar = pull_radar(conn,nsite,ndate)
else:
   # check the all image flag. If set, find all scans and clean. 
   if n_all is True:
      downloads = pull_all(conn,nsite,ndate,outdir=outdir)
   # Call the radar image. 
   else:
      # Read in the radar
      radar = pull_radar(conn,nsite,ndate,outdir=outdir)

#------------------------------------------------------------------------------
# IV) END OF PROGRAM 
#------------------------------------------------------------------------------

# define the ending time of the program for testing purposes 
t1 = time.time()

# print out run time
x = (t1-t0)/60.
print("Run Time: ", x, ' min')

