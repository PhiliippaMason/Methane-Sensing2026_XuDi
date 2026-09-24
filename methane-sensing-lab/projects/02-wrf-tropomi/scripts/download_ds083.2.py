#!/usr/bin/env python
#################################################################
# Python Script to retrieve 13 online Data files of 'ds083.2',
# total 20.92G. This script uses 'requests' to download data.
#
# Highlight this script by Select All, Copy and Paste it into a file;
# make the file executable and run it on command line.
#
# You need pass in your password as a parameter to execute
# this script; or you can set an environment variable RDAPSWD
# if your Operating System supports it.
#
# Contact rdahelp@ucar.edu (RDA help desk) for further assistance.
#################################################################


import sys
import os
import requests


def check_file_status(filepath, filesize):
    sys.stdout.write("\r")
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    percent_complete = (size / filesize) * 100
    sys.stdout.write("%.3f %s" % (percent_complete, "% Completed"))
    sys.stdout.flush()


# Try to get password
# if len(sys.argv) < 2 and 'RDAPSWD' not in os.environ:
#     try:
#         import getpass
#         input = getpass.getpass
#     except:
#         try:
#             input = raw_input
#         except:
#             pass
#     pswd = input('Password: ')
# else:
#     try:
#         pswd = sys.argv[1]
#     except:
#         pswd = os.environ['RDAPSWD']
if len(sys.argv) < 2 and "RDAPSWD" not in os.environ:
    import getpass

    input = getpass.getpass
    pswd = input("Password: ")
else:
    pswd = sys.argv[1]

url = "https://rda.ucar.edu/cgi-bin/login"
values = {"email": "dx522@ic.ac.uk", "passwd": pswd, "action": "login"}
# Authenticate
ret = requests.post(url, data=values)
if ret.status_code != 200:
    print("Bad Authentication")
    print(ret.text)
    exit(1)
dspath = "https://rda.ucar.edu/dsrqst/XU664066/"
filelist = [
    "TarFiles/fnl_20200101_00-20200129_12_00.grib2.tar",
    "TarFiles/fnl_20200129_18-20200227_00_00.grib2.tar",
    "TarFiles/fnl_20200227_06-20200326_12_00.grib2.tar",
    "TarFiles/fnl_20200326_18-20200423_12_00.grib2.tar",
    "TarFiles/fnl_20200423_18-20200521_12_00.grib2.tar",
    "TarFiles/fnl_20200521_18-20200618_06_00.grib2.tar",
    "TarFiles/fnl_20200618_12-20200716_00_00.grib2.tar",
    "TarFiles/fnl_20200716-20200813_06_00.grib2.tar",
    "TarFiles/fnl_20200813_12-20200911_00_00.grib2.tar",
    "TarFiles/fnl_20200911_06-20201009_12_00.grib2.tar",
    "TarFiles/fnl_20201009_18-20201107_00_00.grib2.tar",
    "TarFiles/fnl_20201107_06-20201205_00_00.grib2.tar",
    "TarFiles/fnl_20201205_06-20210101_00_00.grib2.tar",
]
for file in filelist:
    filename = dspath + file
    file_base = os.path.basename(file)
    print("Downloading", file_base)
    req = requests.get(filename, cookies=ret.cookies, allow_redirects=True, stream=True)
    filesize = int(req.headers["Content-length"])
    with open(file_base, "wb") as outfile:
        chunk_size = 1048576
        for chunk in req.iter_content(chunk_size=chunk_size):
            outfile.write(chunk)
            if chunk_size < filesize:
                check_file_status(file_base, filesize)
    check_file_status(file_base, filesize)
    print()
