#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:52:05 2024

@author: rpoitevineau
"""

# This is a test for calling this file from a command line
# The line I use is:
#python test_caller_from_cmdl.py LOFAR gauss4_t201806301100_SBL180.MS -o 4Gauss_t1_sigma095 -n 2000 -f 4.26 -p 15

import bipp_custom_caller as bcc


a = bcc.bipp_custom()

a.display_config()

a.dirty()
