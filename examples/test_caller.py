#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:52:05 2024

@author: rpoitevineau
"""

import bipp_custom_caller as bcc



a = bcc(telescope="LOFAR", 
    ms_file="gauss4_t201806301100_SBL180.MS", 
    output="4Gauss_t1_sigma095", 
    npix=2000, 
    fov=4.26,  
    partition=15, 
    use_cmdline=False)

a.display_config()

a.dirty()
