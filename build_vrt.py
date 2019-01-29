# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:42:09 2018

@author: Erik
"""

import subprocess

path_vrt = VRT_PATH

for year in range(2006,2018,1):
   command=["gdalbuildvrt", path_vrt+'vrt_'+str(year)+".vrt", '-input_file_list', path_vrt+str(year)+'.txt']       
   subprocess.call(command)