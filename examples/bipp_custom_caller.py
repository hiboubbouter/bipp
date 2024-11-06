#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 17:16:05 2024

@author: rpoitevineau
"""


import argparse
from tqdm import tqdm as ProgressBar
import astropy.units as u
import astropy.coordinates as coord
import astropy.io.fits as fits
import astropy.wcs as awcs
import astropy.time as atime
import bipp.imot_tools.io.s2image as s2image
import numpy as np
import scipy.constants as constants
import bipp
import bipp.beamforming as beamforming
import bipp.gram as bb_gr
import bipp.parameter_estimator as bb_pe
import bipp.source as source
import bipp.instrument as instrument
import bipp.frame as frame
import bipp.statistics as statistics
import bipp.measurement_set as measurement_set
import time as tt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import subprocess
import sys
from tqdm import tqdm









class bipp_custom:
    
    def __init__(self, telescope=None, ms_file=None, output=None, npix=None, fov=None, nlevel=None, 
                 clustering=None, partition=None, channel=None, timestep=None, wsclean=None, 
                 grid=None, column=None, eps=None, use_cmdline=True):
        
        # If the arguments are to be taken from the command line
        if use_cmdline:
            parser = argparse.ArgumentParser()

            # Define command-line arguments
            parser.add_argument("telescope", type=str, help="Name of telescope. Currently SKALow, LOFAR and MWA accepted.")
            parser.add_argument("ms_file", type=str, help="Path to .ms file.")
            parser.add_argument("-o", "--output", type=str, help="Name of output file.")
            parser.add_argument("-n", "--npix", type=int, help="Number of pixels in output image.")
            parser.add_argument("-f", "--fov", type=float, help="Field of View of image (in degrees).")
            parser.add_argument("-l", "--nlevel", type=int, help="Number of Bluebild energy levels.")
            parser.add_argument("-b", "--clustering", help="Clustering (Boolean/List of eigenvalue bin edges separated by commas)")
            parser.add_argument("-p", "--partition", type=int, help="Number of partitions for nuFFT.")
            parser.add_argument("-c", "--channel", help="Channels to include in analysis. Give 2 integers separated by a comma: start,end(exclusive).")
            parser.add_argument("-t", "--timestep", help="Timesteps to include in analysis. Give 2 integers separated by a comma: start,end(exclusive).")
            parser.add_argument("-w", "--wsclean", help="Path to WSClean .fits file (For comparison)")
            parser.add_argument("-g", "--grid", type=str, help="Which grid to use (ms, wsclean, RA,Dec)")
            parser.add_argument("--column", type=str, help="Which column from the measurement set file to image. Eg: DATA, CORRECTED_DATA, MODEL_DATA")
            parser.add_argument("-e", "--eps", help="Error Tolerance of the nuFFT.")
            
            args = parser.parse_args()

            # Assigning command-line values or default constructor values
            self.telescope = telescope if telescope else args.telescope
            self.ms_file = ms_file if ms_file else args.ms_file
            self.output = output if output else args.output
            self.npix = npix if npix else args.npix
            self.fov = fov if fov else args.fov
            self.nlevel = nlevel if nlevel else args.nlevel
            self.clustering = clustering if clustering else args.clustering
            self.partition = partition if partition else args.partition
            self.channel = channel if channel else args.channel
            self.timestep = timestep if timestep else args.timestep
            self.wsclean = wsclean if wsclean else args.wsclean
            self.grid = grid if grid else args.grid
            self.column = column if column else args.column
            self.eps = eps if eps else args.eps

        # If not using command-line arguments
        else:
            self.telescope = telescope
            self.ms_file = ms_file
            self.output = output
            self.npix = npix
            self.fov = fov
            self.nlevel = nlevel
            self.clustering = clustering
            self.partition = partition
            self.channel = channel
            self.timestep = timestep
            self.wsclean = wsclean
            self.grid = grid
            self.column = column
            self.eps = eps
        
        # Post-initialization checks and logic from your original code
        self.setup_telescope()
        self.setup_defaults()
        self.setup_clustering()
        self.setup_partition()
        self.setup_channels()
        self.setup_grid()
        
###########################
    def setup_telescope(self):
        if self.telescope.lower() == "skalow":
            self.ms = measurement_set.SKALowMeasurementSet(self.ms_file)
            self.N_station, self.N_antenna = 512, 512
        #elif self.telescope.lower() == "redundant":
        #    self.ms = measurement_set.GenericMeasurementSet(self.ms_file)
        #    self.N_station = self.ms.AntennaNumber()
        #    self.N_antenna = self.N_station
        elif self.telescope.lower() == "mwa":
            self.ms = measurement_set.MwaMeasurementSet(self.ms_file)
            self.N_station, self.N_antenna = 128, 128
        elif self.telescope.lower() == "lofar":
            self.N_station, self.N_antenna = 37, 37
            self.ms = measurement_set.LofarMeasurementSet(self.ms_file, N_station=self.N_station, station_only=True)
        else:
            raise NotImplementedError(f"A measurement set class for the telescope {self.telescope} has not been implemented yet.")
        print(f"N_station:{self.N_station} , N_antenna:{self.N_antenna}")
        
###########################        
    def setup_defaults(self):
        if self.output is None:
            self.output = "test"
        if self.npix is None:
            self.npix = 2000
        if self.fov is None:
            self.fov = np.deg2rad(7)
        else:
            self.fov = np.deg2rad(self.fov)
        if self.nlevel is None:
            self.nlevel = 4
        if self.eps is None:
            self.eps = 1e-3
        if (self.column==None):
            self.column = "DATA" 
 ###########################       
    def setup_clustering(self):
        if self.clustering is None:
            self.clustering = True
            self.clusteringBool = True
        else:
            try:
                clusterEdges = np.array(self.clustering.split(","), dtype=np.float32)
                self.clusteringBool = False
                binStart = clusterEdges[0]
                clustering = []
                for binEdge in clusterEdges[1:]:
                    binEnd = binEdge
                    clustering.append([binEnd, binStart])
                    binStart = binEnd
                self.clustering = np.asarray(clustering, dtype=np.float32)
            except:
                self.clustering = bool(self.clustering)
                self.clusteringBool = True
                
 ###########################               
    def setup_partition(self):
        if self.partition is None:
            self.partition = 1
            
###########################            
    def setup_channels(self):
        if self.timestep is None:
            self.timeStart = 0
            self.timeEnd = -1
        else:
            [self.timeStart, self.timeEnd] = np.array(self.timestep.split(","), dtype=np.int32)

        if self.channel is None:
            self.channelStart = 0
            self.channelEnd = -1
            self.nChannel = self.ms.channels["FREQUENCY"].shape[0]
        else:
            [self.channelStart, self.channelEnd] = np.array(self.channel.split(","), dtype=np.int32)
            self.nChannel = self.channelEnd - self.channelStart
            
###########################            
    def setup_grid(self):
        if self.grid is None:
            self.ms_fieldcenter = True
            self.grid = "ms"
        elif self.grid.lower() == "ms":
            self.ms_fieldcenter = True
        elif self.grid.lower() == "wsclean":
            self.ms_fieldcenter = False
        elif len(self.grid.split(",")) == 2:
            self.ms_fieldcenter = False
            [self.RA, self.Dec] = np.array(self.grid.split(","), dtype=np.float32)
        else:
            raise NotImplementedError("Only wsclean, ms and RA,Dec (degrees) grids have been defined so far.")
            
###########################
    def display_config(self):
        print(f"Telescope Name:{self.telescope}")
        print(f"MS file:{self.ms_file}")
        print(f"Output Name:{self.output}")
        print(f"N_Pix:{self.npix} pixels")
        print(f"FoV:{np.rad2deg(self.fov)} deg")
        print(f"N_level:{self.nlevel} levels")
        print(f"Clustering Bool:{self.clusteringBool}")
        kmeans = "kmeans"
        print(f"Clustering:{kmeans if self.clusteringBool else self.clustering}")
        print(f"Grid:{self.grid}")
        print(f"MS Field Center:{self.ms_fieldcenter}")
        print(f"MS Channel Start:{self.channelStart} Channel End:{self.channelEnd}")
        print(f"MS Timestep Start:{self.timeStart} Timestep End:{self.timeEnd}")
        print(f"WSClean Path:{self.wsclean}")
        print(f"Partitions:{self.partition}")
        print(f"MS Column Used: {self.column}")
        print(f"nuFFT tolerance: {self.eps}")

###########################
    def observations_set_up(self):
        
        self.sampling = 50
        self.precision = 'double'
        self.ctx = bipp.Context("GPU")
        self.filter_tuple = ['lsq','std']
        self.std_img_flag = True 
        self.plotList= np.array([3,])
        self.outputCustomFitsFile = True
        
        if (self.channelEnd - self.channelStart == 1): 
            self.frequency = self.ms.channels["FREQUENCY"][self.channelStart:self.channelEnd]
            print ("Single channel mode.")
            self.channel_id = np.arange(self.channelStart, self.channelEnd, dtype = np.int32)
                
        else:
            self.frequency = self.ms.channels["FREQUENCY"][self.channelStart] + (self.ms.channels["FREQUENCY"][self.channelEnd] - self.ms.channels["FREQUENCY"][self.channelStart])/2
            self.channel_id = np.arange(self.channelStart, self.nChannel, dtype = np.int32)
            #print (f"Multi-channel mode with {channelEnd - channelStart}channels.")
    
        self.wl = constants.speed_of_light / self.frequency.to_value(u.Hz)
        #print (f"wl:{wl}; f: {frequency}")
    
        if (self.grid == "ms"):
            self.field_center = self.ms.field_center
            print ("Self generated grid used based on ms fieldcenter")
            
        elif (self.grid =="wsclean"): 
            with fits.open(self.wsclean, mode="readonly", memmap=True, lazy_load_hdus=True) as hdulist:
                cl_WCS = awcs.WCS(hdulist[0].header)
                cl_WCS = cl_WCS.sub(['celestial'])
                cl_WCS = cl_WCS.slice((slice(None, None, self.sampling), slice(None, None, self.sampling)))
    
            self.field_center = self.ms.field_center
    
            self.width_px, self.height_px= 2*cl_WCS.wcs.crpix 
            self.cdelt_x, self.cdelt_y = cl_WCS.wcs.cdelt 
            self.FoV = np.deg2rad(abs(self.cdelt_x*self.width_px))
            print ("WSClean Grid used.")
            
        elif (len(self.grid.split(",")) == 2):
            self.field_center = coord.SkyCoord(ra=self.RA * u.deg, dec=self.Dec * u.deg, frame="icrs")
            print ("Self generated grid used based on user fieldcenter.")
            
        else:
            raise NotImplementedError ("This gridstyle has not been implemented.")

###########################
    def gridder(self):
        self.lmn_grid, self.xyz_grid = frame.make_grids(self.npix, self.fov, self.field_center)

###########################
    def gram(self):
        self.gram = bb_gr.GramBlock(self.ctx)

###########################
    def set_up_nufft(self, weighting):
        self.opt = bipp.NufftSynthesisOptions()
        # Set the tolerance for NUFFT, which is the maximum relative error.
        self.opt.set_tolerance(self.eps)
        # Set the maximum number of data packages that are processed together after collection.
        # A larger number increases memory usage, but usually improves performance.
        # If set to "None", an internal heuristic will be used.
        self.opt.set_collect_group_size(None)
        # Set the domain splitting methods for image and uvw coordinates.
        # Splitting decreases memory usage, but may lead to lower performance.
        # Best used with a wide spread of image or uvw coordinates.
        # Possible options are "grid", "none" or "auto"
        #opt.set_local_image_partition(bipp.Partition.grid([1,1,1]))
        #opt.set_local_image_partition(bipp.Partition.none()) # Commented out
        #opt.set_local_uvw_partition(bipp.Partition.none()) # Commented out
        self.opt.set_local_image_partition(bipp.Partition.grid([1,1,1]))
        self.opt.set_local_uvw_partition(bipp.Partition.grid([self.partition,self.partition,1]))
        
        if weighting == 'uniform':
            self.opt.set_normalize_image_by_nvis(False)
            self.opt.set_normalize_image(False)
        elif weighting == 'natural':
            self.opt.set_normalize_image_by_nvis(True)
            self.opt.set_normalize_image(True)
        #opt.set_local_image_partition(bipp.Partition.auto())
        #opt.set_local_uvw_partition(bipp.Partition.auto())



########################################################################################
    def parameter_estimation(self):
    
        ########################################################################################
        ### Intensity Field  Parameter Estimation ##############################################
        ########################################################################################
        #self.pe_t = self.tt.time()
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@ PARAMETER ESTIMATION @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
        num_time_steps = 0
        n = 1
        I_est = bb_pe.ParameterEstimator(self.nlevel, sigma=1, ctx=self.ctx, fne=False)
        for t, f, S in ProgressBar(
                self.ms.visibilities(channel_id=self.channel_id, time_id=slice(self.timeStart, self.timeEnd, n), column=self.column)
        ):
          
            wl = constants.speed_of_light / f.to_value(u.Hz)
            XYZ = self.ms.instrument(t)
        
            W = self.ms.beamformer(XYZ, wl)
            self.G = self.gram(XYZ, W, wl)
            S, _ = measurement_set.filter_data(S, W)
            I_est.collect(wl, S.data, W.data, XYZ.data)
            num_time_steps +=1
        
        self.intervals = I_est.infer_parameters()
        self.fi = bipp.filter.Filter(lsq=self.intervals, std=self.intervals)



########################################################################################
    def def_imaging(self):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ IMAGING @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
        self.imager = bipp.NufftSynthesis(
            self.ctx,
            self.opt,
            self.fi.num_images(),
            self.lmn_grid[0],
            self.lmn_grid[1],
            self.lmn_grid[2],
           self. precision,
        )

###########################
    def get_uv(self, n):
        uu = []
        vv = []
        for t, f, S in ProgressBar(
                self.ms.visibilities(channel_id=self.channel_id, time_id=slice(self.timeStart, self.timeEnd, n), column=self.column)
        ):
            wl = constants.speed_of_light / f.to_value(u.Hz)
            UVW_baselines_t = self.ms.instrument.baselines(t, uvw=True, field_center=self.ms.field_center)
            uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
            ut, vt, wt = uvw.T
            uu.extend(ut)
            vv.extend(vt)
        self.uu = np.array(uu)
        self.vv = np.array(vv)
        
########################### 
    def uv_grid(self, N, du, u, v):
        grid_start = -N//2 * du
        grid_end = N//2 * du
        xedges = np.linspace(grid_start, grid_end, N+1)
        yedges = np.linspace(grid_start, grid_end, N+1)
        counts, _, _ = np.histogram2d(u, v, bins=[xedges, yedges])
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_counts = np.true_divide(1, counts)
            inv_counts[~np.isfinite(inv_counts)] = 0
        return xedges, yedges, counts, inv_counts
   
###########################
    def weighting(self, u, v, xedges, yedges, counts):
        x_idx = np.digitize(u, xedges) - 1
        y_idx = np.digitize(v, yedges) - 1
        x_idx = np.clip(x_idx, 0, len(xedges)-2)# N-1)
        y_idx = np.clip(y_idx, 0, len(yedges)-2)#N-1)
        assigned_weights = counts[x_idx, y_idx]
        for i in range(len(u)):
            if u[i]<xedges[0] or u[i]>xedges[-1] or v[i]<yedges[0] or v[i]>yedges[-1]:
                assigned_weights[i] = 0.0
        return assigned_weights  
    
###############################
    def imaging(self, weighting='uniform', n=1):
        ###
        if weighting == 'uniform':
            print('Uniform weighting')
            print('du = ', self.du)
            print('nbr non empty cells = ', self.non_empty_cells)
            temp = 0
            for t, f, S in ProgressBar(
                    self.ms.visibilities(channel_id=self.channel_id, time_id=slice(self.timeStart, self.timeEnd, n), column=self.column)
            ):
                
                wl = constants.speed_of_light / f.to_value(u.Hz)
                XYZ = self.ms.instrument(t)
                W = self.ms.beamformer(XYZ, wl)
                S, W = measurement_set.filter_data(S, W)
        
                UVW_baselines_t = self.ms.instrument.baselines(t, uvw=True, field_center=self.ms.field_center)
                uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
                if np.allclose(S.data, np.zeros(S.data.shape)):
                    continue
                new_S = S.data.T.reshape(-1, order="F")
                ut, vt, wt = uvw.T
                w_okay = self.weighting(ut, vt, self.xedges, self.yedges, self.inv_counts)
                temp += np.sum(w_okay)
                # new_S = np.full(new_S.shape, 1 + 0j)
                new_S = new_S*w_okay
                new_S = new_S.reshape((S.data.shape[0], S.data.shape[0]), order="F").T
                self.imager.collect(wl, self.fi, new_S, W.data, XYZ.data, uvw)
            print('sum of the weights = ', temp)
        ###
        if weighting == 'natural':
                print('Natural weighting')
                for t, f, S in ProgressBar(
                        self.ms.visibilities(channel_id=self.channel_id, time_id=slice(self.timeStart, self.timeEnd, n), column=self.column)
                ):
                    
                    wl = constants.speed_of_light / f.to_value(u.Hz)
                    XYZ = self.ms.instrument(t)
                    W = self.ms.beamformer(XYZ, wl)
                    S, W = measurement_set.filter_data(S, W)
            
                    UVW_baselines_t = self.ms.instrument.baselines(t, uvw=True, field_center=self.ms.field_center)
                    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
                    if np.allclose(S.data, np.zeros(S.data.shape)):
                        continue
                    self.imager.collect(wl, self.fi, S.data, W.data, XYZ.data, uvw)
                    
###############################
    def save_fits(self):
        if (self.outputCustomFitsFile):

            w = awcs.WCS(naxis=2)

            
            w.wcs.crpix = np.array([self.npix//2 + 1, self.npix//2 + 1])
            w.wcs.cdelt = np.array([-np.rad2deg(self.fov)/self.npix, np.rad2deg(self.fov)/self.npix])
            w.wcs.crval = np.array([self.field_center.ra.deg, self.field_center.dec.deg])
            w.wcs.ctype = ["RA---SIN", "DEC--SIN"]

            header = w.to_header()
            hdu =fits.PrimaryHDU(np.fliplr(self.I_lsq_eq_summed.data),header=header)

            #hdu.header['SIMPLE'] = "T" # fits compliant format
            if (self.precision.lower()=='double'):
                hdu.header['BITPIX']=-64 # double precision float
            elif (self.precision.lower()=='single'):
                hdu.header['BITPIX']=-32 # single precision float
            hdu.header['NAXIS'] = 2 # Number of axes - 2 for image data, 3 for data cube
            hdu.header['NAXIS1'] = self.I_lsq_eq_summed.shape[-2]
            hdu.header['NAXIS2'] = self.I_lsq_eq_summed.shape[-1]
            #shdu.header['EXTEND'] = "T" # Fits data set may contain extensions
            hdu.header['BSCALE'] = 1 # scale to be multiplied by the data array values when reading the FITS file
            hdu.header['BZERO'] = 0 # zero offset to be added to the data array values when reading the FITS file
            hdu.header['BUNIT'] = 'Jy/Beam' # Units of the data array
            hdu.header['BTYPE'] = 'Intensity'
            hdu.header['ORIGIN'] = "BIPP"
            #hdu.header['HISTORY'] = sys.argv[:]

            hdu.writeto(f"{self.output}_summed.fits", overwrite=True)

            for i in np.arange(self.nlevel):
                hdu =fits.PrimaryHDU(np.fliplr(self.I_lsq_eq.data[i, :, :]),header=header)

                #hdu.header['SIMPLE'] = 'T' # fits compliant format
                if (self.precision.lower()=='double'):
                    hdu.header['BITPIX']=-64 # double precision float
                elif (self.precision.lower()=='single'):
                    hdu.header['BITPIX']=-32 # single precision float
                hdu.header['NAXIS'] = 2 # Number of axes - 2 for image data, 3 for data cube
                hdu.header['NAXIS1'] = self.I_lsq_eq_summed.shape[-2]
                hdu.header['NAXIS2'] = self.I_lsq_eq_summed.shape[-1]
                #shdu.header['EXTEND'] = "T" # Fits data set may contain extensions
                hdu.header['BSCALE'] = 1 # scale to be multiplied by the data array values when reading the FITS file
                hdu.header['BZERO'] = 0 # zero offset to be added to the data array values when reading the FITS file
                hdu.header['BUNIT'] = 'Jy/Beam' # Units of the data array
                hdu.header['BTYPE'] = 'Intensity'
                hdu.header['ORIGIN'] = "BIPP"
                #hdu.header['HISTORY'] = sys.argv[:]

                hdu.writeto(f"{self.output}_lvl{i}.fits", overwrite=True)

        else:
            self.I_lsq_eq_summed.to_fits(f"{self.output}_summed.fits")
            self.I_lsq_eq.to_fits(f"{self.output}_lvls.fits")


########################################################################################
    def dirty(self, weighting='uniform', n=1, save=True):
        
        self.observations_set_up()
        self.gridder()
        self.gram()
        self.set_up_nufft(weighting)
        self.parameter_estimation()
        self.def_imaging()
        
        if weighting == 'uniform':
            self.get_uv(n)
            self.du = 1/self.fov * self.wl
            self.xedges, self.yedges, self.counts, self.inv_counts  = self.uv_grid(self.npix, self.du, self.uu, self.vv)
            self.non_empty_cells = float(np.count_nonzero(self.counts))
        
            self.imaging(weighting, n)
            images = self.imager.get().reshape((-1, self.npix, self.npix))
            self.lsq_image = self.fi.get_filter_images("lsq", images)
            self.std_image = self.fi.get_filter_images("std", images)
            
            self.lsq_image.data = self.lsq_image.data / self.non_empty_cells
            self.std_image.data = self.std_image.data / self.non_empty_cells
            
            self.I_lsq_eq = s2image.Image(self.lsq_image, self.xyz_grid)
            self.I_std_eq = s2image.Image(self.std_image, self.xyz_grid)
            self.I_lsq_eq_summed = s2image.Image(self.lsq_image.reshape(self.nlevel+1,self.lsq_image.shape[-2], self.lsq_image.shape[-1]).sum(axis = 0), self.xyz_grid)
            self.I_std_eq_summed = s2image.Image(self.std_image.reshape(self.nlevel+1,self.std_image.shape[-2], self.std_image.shape[-1]).sum(axis = 0), self.xyz_grid)

            self.I_lsq_eq_summed.data = self.I_lsq_eq_summed.data / self.non_empty_cells
            self.I_std_eq_summed.data = self.I_std_eq_summed.data / self.non_empty_cells
        
        if weighting == 'natural':
        
            self.imaging(weighting, n)
            images = self.imager.get().reshape((-1, self.npix, self.npix))
            self.lsq_image = self.fi.get_filter_images("lsq", images)
            self.std_image = self.fi.get_filter_images("std", images)
            self.I_lsq_eq = s2image.Image(self.lsq_image, self.xyz_grid)
            self.I_std_eq = s2image.Image(self.std_image, self.xyz_grid)
            self.I_lsq_eq_summed = s2image.Image(self.lsq_image.reshape(self.nlevel+1,self.lsq_image.shape[-2], self.lsq_image.shape[-1]).sum(axis = 0), self.xyz_grid)
            self.I_std_eq_summed = s2image.Image(self.std_image.reshape(self.nlevel+1,self.std_image.shape[-2], self.std_image.shape[-1]).sum(axis = 0), self.xyz_grid)
            
            
        if save == True:
            self.save_fits()

########################################################################################
    def psf(self, weighting, n=1, save=True):
        
        self.observations_set_up()
        self.gridder()
        self.gram()
        self.set_up_nufft(weighting)
        self.parameter_estimation()
        self.def_imaging()
        
        
        if weighting == 'uniform':
            self.get_uv(n)
            self.du = 1/self.fov * self.wl
            self.xedges, self.yedges, self.counts, self.inv_counts  = self.uv_grid(self.npix, self.du, self.uu, self.vv)
            self.non_empty_cells = float(np.count_nonzero(self.counts))
        
            self.imaging_psf(weighting, n)
            images = self.imager.get().reshape((-1, self.npix, self.npix))
            self.lsq_image = self.fi.get_filter_images("lsq", images)
            self.std_image = self.fi.get_filter_images("std", images)
            
            self.I_lsq_eq = s2image.Image(self.lsq_image, self.xyz_grid)
            self.I_std_eq = s2image.Image(self.std_image, self.xyz_grid)
            self.I_lsq_eq_summed = s2image.Image(self.lsq_image.reshape(self.nlevel+1,self.lsq_image.shape[-2], self.lsq_image.shape[-1]).sum(axis = 0), self.xyz_grid)
            self.I_std_eq_summed = s2image.Image(self.std_image.reshape(self.nlevel+1,self.std_image.shape[-2], self.std_image.shape[-1]).sum(axis = 0), self.xyz_grid)

            self.I_lsq_eq_summed.data = self.I_lsq_eq_summed.data / self.non_empty_cells
            self.I_std_eq_summed.data = self.I_std_eq_summed.data / self.non_empty_cells
            
        if weighting == 'natural':
        
            self.imaging_psf(weighting, n)
            images = self.imager.get().reshape((-1, self.npix, self.npix))
            self.lsq_image = self.fi.get_filter_images("lsq", images)
            self.std_image = self.fi.get_filter_images("std", images)
            self.I_lsq_eq = s2image.Image(self.lsq_image, self.xyz_grid)
            self.I_std_eq = s2image.Image(self.std_image, self.xyz_grid)
            self.I_lsq_eq_summed = s2image.Image(self.lsq_image.reshape(self.nlevel+1,self.lsq_image.shape[-2], self.lsq_image.shape[-1]).sum(axis = 0), self.xyz_grid)
            self.I_std_eq_summed = s2image.Image(self.std_image.reshape(self.nlevel+1,self.std_image.shape[-2], self.std_image.shape[-1]).sum(axis = 0), self.xyz_grid)
            
            
            self.save_fits()
        
        
        
        
        
        
    def imaging_psf(self, weighting='uniform',n=1):
            ###
            if weighting == 'uniform':
                print('Uniform weighting PSF')
                print('du = ', self.du)
                print('nbr non empty cells = ', self.non_empty_cells)
                temp = 0
                for t, f, S in ProgressBar(
                        self.ms.visibilities(channel_id=self.channel_id, time_id=slice(self.timeStart, self.timeEnd, n), column=self.column)
                ):
                    
                    wl = constants.speed_of_light / f.to_value(u.Hz)
                    XYZ = self.ms.instrument(t)
                    W = self.ms.beamformer(XYZ, wl)
                    S, W = measurement_set.filter_data(S, W)
            
                    UVW_baselines_t = self.ms.instrument.baselines(t, uvw=True, field_center=self.ms.field_center)
                    uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
                    if np.allclose(S.data, np.zeros(S.data.shape)):
                        continue
                    new_S = S.data.T.reshape(-1, order="F")
                    ut, vt, wt = uvw.T
                    w_okay = self.weighting(ut, vt, self.xedges, self.yedges, self.inv_counts)
                    temp += np.sum(w_okay)
                    new_S = np.full(new_S.shape, 1 + 0j)
                    new_S = new_S*w_okay
                    new_S = new_S.reshape((S.data.shape[0], S.data.shape[0]), order="F").T
                    self.imager.collect(wl, self.fi, new_S, W.data, XYZ.data, uvw)
                print('sum of the weights = ', temp)
            ###
            if weighting == 'natural':
                    print('Natural weighting PSF')
                    for t, f, S in ProgressBar(
                            self.ms.visibilities(channel_id=self.channel_id, time_id=slice(self.timeStart, self.timeEnd, n), column=self.column)
                    ):
                        
                        wl = constants.speed_of_light / f.to_value(u.Hz)
                        XYZ = self.ms.instrument(t)
                        W = self.ms.beamformer(XYZ, wl)
                        S, W = measurement_set.filter_data(S, W)
                
                        UVW_baselines_t = self.ms.instrument.baselines(t, uvw=True, field_center=self.ms.field_center)
                        uvw = frame.reshape_and_scale_uvw(wl, UVW_baselines_t)
                        if np.allclose(S.data, np.zeros(S.data.shape)):
                            continue
                        self.imager.collect(wl, self.fi, np.full(S.shape, 1 + 0j, W.data, XYZ.data, uvw)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
