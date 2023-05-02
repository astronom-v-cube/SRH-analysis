#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 11:44:53 2022

@author: sergey_lesovoi
"""

from pyuvdata import UVData
from srhFitsFile612 import SrhFitsFile
import numpy as NP
from astropy.time import Time
import astropy.coordinates as COORD
from astropy.coordinates import EarthLocation
from astropy import units as u
from datetime import timedelta

class SrhUVData(UVData):
    """
    """
    def __init__(self):
        self.srhLat = 51.759
        self.srhLon = 102.217
        self.srhAlt = 799.
        self.srhEarthLocation = EarthLocation(lon=self.srhLon * u.deg, lat=self.srhLat * u.deg, height=self.srhAlt * u.m)
        self.srhLocation = self.srhEarthLocation.T.to_value()
        self.srhLat = NP.deg2rad(self.srhLat)
        self.srhLon = NP.deg2rad(self.srhLon)
        self.phi_operator = NP.array([
            [-NP.sin(self.srhLat), 0., NP.cos(self.srhLat)],
            [0., 1., 0.],
            [NP.cos(self.srhLat), 0., NP.sin(self.srhLat)]
            ])
        super().__init__()
        
    def baseline2uvw(self, hourAngle, declination, antenna0, antenna1):
        if ((antenna0 >= 1 and antenna0 <= 128) and (antenna1 >= 129 and antenna1 <= 192)):
            base = NP.array([antenna1 - 128.5,antenna0 - 64.5,0.])
        elif ((antenna1 >= 1 and antenna1 <= 128) and (antenna0 >= 129 and antenna0 <= 192)):
            base = NP.array([antenna0 - 128.5,antenna1 - 64.5,0.])
        elif ((antenna0 >= 1 and antenna0 <= 128) and (antenna1 >= 1 and antenna1 <= 128)):
            base = NP.array([0.,antenna0 - antenna1,0.])
        elif ((antenna0 >= 129 and antenna0 <= 192) and (antenna1 >= 129 and antenna1 <= 192)):
            base = NP.array([antenna0 - antenna1,0.,0.])
        
        base *= 4.9;
        
        uvw_operator = NP.array([
            [ NP.sin(hourAngle),		 NP.cos(hourAngle),		0.	  ],
            [-NP.sin(declination)*NP.cos(hourAngle),  NP.sin(declination)*NP.sin(hourAngle), NP.cos(declination)], 
            [ NP.cos(declination)*NP.cos(hourAngle), -NP.cos(declination)*NP.sin(hourAngle), NP.sin(declination)]  
            ])
    

        return NP.dot(uvw_operator, NP.dot(self.phi_operator, base))

    def write_uvfits(self, srcName, dstName, lastVisibility=8192, frequency=0, scan=0, calibrating=True, uvSize=1024, averaging=0):
        srhFits = SrhFitsFile(srcName, uvSize)
        srhFits.setFrequencyChannel(frequency)
        srhFits.getHourAngle(scan)
        if (calibrating):
            srhFits.calibrate(frequency)
        srhFits.vis2uv(scan, average=averaging)
        visRcp = srhFits.visRcp[frequency,scan,0:lastVisibility].copy()
        visLcp = srhFits.visLcp[frequency,scan,0:lastVisibility].copy()
        scanTime = Time(srhFits.dateObs.split('T')[0] + 'T' + str(timedelta(seconds=srhFits.freqTime[frequency,scan])))
        
        coords = COORD.get_sun(scanTime)
        
        for vis in range(lastVisibility):
            i = vis // 128
            j = vis % 128 
            visLcp[vis] *= NP.exp(1j*(-(srhFits.ewAntPhaLcp[frequency, j]+srhFits.ewLcpPhaseCorrection[frequency, j]) + (srhFits.sAntPhaLcp[frequency, i] + srhFits.sLcpPhaseCorrection[frequency, i])))
            visRcp[vis] *= NP.exp(1j*(-(srhFits.ewAntPhaRcp[frequency, j]+srhFits.ewRcpPhaseCorrection[frequency, j]) + (srhFits.sAntPhaRcp[frequency, i] + srhFits.sRcpPhaseCorrection[frequency, i])))
            visLcp[vis] /= (srhFits.ewAntAmpLcp[frequency, j] * srhFits.sAntAmpLcp[frequency, i])
            visRcp[vis] /= (srhFits.ewAntAmpRcp[frequency, j] * srhFits.sAntAmpRcp[frequency, i])
        
        self.ant_1_array = srhFits.antennaA[0:lastVisibility]
        self.ant_2_array = srhFits.antennaB[0:lastVisibility]
        self.antenna_names = srhFits.antennaNames
        self.antenna_numbers = NP.array(list(map(int,srhFits.antennaNumbers)))
        self.Nants_data = NP.union1d(srhFits.antennaA[0:lastVisibility],srhFits.antennaB[0:lastVisibility]).size
        self.Nants_telescope = srhFits.antennaNames.shape[0]
        self.Nfreqs = 1
        self.Npols = 2
        self.Ntimes = 1
        self.Nspws = 1
        self.Nbls = lastVisibility
        self.Nblts = lastVisibility * self.Ntimes
        self.phase_type = 'phased'
        self.phase_center_ra = coords.ra.rad
        self.phase_center_dec = coords.dec.rad
        self.phase_center_app_ra = NP.full(self.Nblts,coords.ra.rad)
        self.phase_center_app_dec = NP.full(self.Nblts,coords.dec.rad)
        self.phase_center_epoch = 2000.0
        self.channel_width = 1e7
        self.freq_array = NP.zeros((1,1))
        self.freq_array[0] = 5.8e9
        self.history = 'SRH'
        self.instrument = 'SRH'
        self.integration_time = NP.full(self.Nblts,0.1)
        self.antenna_diameters = NP.full(self.Nants_telescope,2.)
        self.lst_array = NP.full(self.Nblts,0.)
        self.object_name = 'Sun'
        self.polarization_array = NP.array([-1,-2])
        self.spw_array = [1]
        self.telescope_location = list(self.srhLocation)
        self.telescope_name = 'SRH'
        self.time_array = NP.full(self.Nblts,scanTime.jd)
        self.data_array = NP.zeros((self.Nblts,1,self.Nfreqs,self.Npols),dtype='complex')
        self.data_array[:,0,0,0] = visRcp
        self.data_array[:,0,0,1] = visLcp
            
        self.flag_array = NP.full((self.Nblts,1,self.Nfreqs,self.Npols),False,dtype='bool')
        self.nsample_array = NP.full((self.Nblts,1,self.Nfreqs,self.Npols),1,dtype='float')
        self.vis_units = 'uncalib'

        self.antenna_positions = NP.zeros((self.Nants_telescope,3))
        for ant in NP.arange(0, 128):
            self.antenna_positions[ant] = [0, (ant - 63.5) * 4.9, 0]
        for ant in NP.arange(128, 192):
            self.antenna_positions[ant] =  [(ant - 127.5) * 4.9, 0, 0]
        self.baseline_array = 2048 * (self.ant_2_array + 1) + self.ant_1_array + 1 + 2**16
        self.uvw_array = self.antenna_positions[self.ant_1_array] - self.antenna_positions[self.ant_2_array]

        lst = Time(scanTime,scale='utc', location=self.srhEarthLocation).sidereal_time('mean')
        hourAngle = lst.to('rad').value - self.phase_center_ra
        for vis in range(lastVisibility):
            self.uvw_array[vis] = self.baseline2uvw(hourAngle,coords.dec.rad,self.ant_2_array[vis] + 1, self.ant_1_array[vis] + 1)
        
        super().write_uvfits(dstName,write_lst=False,spoof_nonessential=True,run_check=False)

    def write_uvfits_all(self, srcName, dstName, lastVisibility=8192, frequency=0, calibrating=True, uvSize=1024):
        srhFits = SrhFitsFile(srcName, uvSize)
        srhFits.setFrequencyChannel(frequency)
        srhFits.getHourAngle(0)
        if (calibrating):
            srhFits.calibrate(frequency)
        srhFits.vis2uv(0, average=20)
        visRcp = srhFits.visRcp[frequency,:,0:lastVisibility].copy()
        visLcp = srhFits.visLcp[frequency,:,0:lastVisibility].copy()
        scanTime = Time(srhFits.dateObs.split('T')[0] + 'T' + str(timedelta(seconds=srhFits.freqTime[frequency,0])))
        
        coords = COORD.get_sun(scanTime)
        
        for vis in range(lastVisibility):
            i = vis // 128
            j = vis % 128 
            visLcp[:,vis] *= NP.exp(1j*(-(srhFits.ewAntPhaLcp[frequency, j]+srhFits.ewLcpPhaseCorrection[frequency, j]) + (srhFits.sAntPhaLcp[frequency, i] + srhFits.sLcpPhaseCorrection[frequency, i])))
            visRcp[:,vis] *= NP.exp(1j*(-(srhFits.ewAntPhaRcp[frequency, j]+srhFits.ewRcpPhaseCorrection[frequency, j]) + (srhFits.sAntPhaRcp[frequency, i] + srhFits.sRcpPhaseCorrection[frequency, i])))
            visLcp[:,vis] /= (srhFits.ewAntAmpLcp[frequency, j] * srhFits.sAntAmpLcp[frequency, i])
            visRcp[:,vis] /= (srhFits.ewAntAmpRcp[frequency, j] * srhFits.sAntAmpRcp[frequency, i])
        
        self.ant_1_array = srhFits.antennaA[0:lastVisibility]
        self.ant_2_array = srhFits.antennaB[0:lastVisibility]
        self.antenna_names = srhFits.antennaNames
        self.antenna_numbers = NP.array(list(map(int,srhFits.antennaNumbers)))
        self.Nants_data = NP.union1d(srhFits.antennaA[0:lastVisibility],srhFits.antennaB[0:lastVisibility]).size
        self.Nants_telescope = srhFits.antennaNames.shape[0]
        self.Nfreqs = 1
        self.Npols = 2
        self.Ntimes = 20
        self.Nspws = 1
        self.Nbls = lastVisibility
        self.Nblts = lastVisibility * self.Ntimes
        self.phase_type = 'phased'
        self.phase_center_ra = coords.ra.rad
        self.phase_center_dec = coords.dec.rad
        self.phase_center_app_ra = NP.full(self.Nblts,coords.ra.rad)
        self.phase_center_app_dec = NP.full(self.Nblts,coords.dec.rad)
        self.phase_center_epoch = 2000.0
        self.channel_width = 1e7
        self.freq_array = NP.zeros((1,1))
        self.freq_array[0] = srhFits.freqList[frequency]*1e3
        self.history = 'SRH'
        self.instrument = 'SRH'
        self.integration_time = NP.full(self.Nblts,0.1)
        self.antenna_diameters = NP.full(self.Nants_telescope,2.)
        self.lst_array = NP.full(self.Nblts,0.)
        self.object_name = 'Sun'
        self.polarization_array = NP.array([-1,-2])
        self.spw_array = [1]
        self.telescope_location = list(self.srhLocation)
        self.telescope_name = 'SRH'
        self.time_array = NP.full(self.Nblts,scanTime.jd)
        self.data_array = NP.zeros((self.Nbls,self.Ntimes,self.Nfreqs,self.Npols),dtype='complex')
        self.data_array[:,:,0,0] = visRcp
        self.data_array[:,:,0,1] = visLcp
            
        self.flag_array = NP.full((self.Nblts,1,self.Nfreqs,self.Npols),False,dtype='bool')
        self.nsample_array = NP.full((self.Nblts,1,self.Nfreqs,self.Npols),1,dtype='float')
        self.vis_units = 'uncalib'

        self.antenna_positions = NP.zeros((self.Nants_telescope,3))
        for ant in NP.arange(0, 128):
            self.antenna_positions[ant] = [0, (ant - 63.5) * 4.9, 0]
        for ant in NP.arange(128, 192):
            self.antenna_positions[ant] =  [(ant - 127.5) * 4.9, 0, 0]
        self.baseline_array = 2048 * (self.ant_2_array + 1) + self.ant_1_array + 1 + 2**16
        self.uvw_array = self.antenna_positions[self.ant_1_array] - self.antenna_positions[self.ant_2_array]

        lst = Time(scanTime,scale='utc', location=self.srhEarthLocation).sidereal_time('mean')
        hourAngle = lst.to('rad').value - self.phase_center_ra
        for vis in range(lastVisibility):
            self.uvw_array[vis] = self.baseline2uvw(hourAngle,coords.dec.rad,self.ant_2_array[vis] + 1, self.ant_1_array[vis] + 1)
        
        super().write_uvfits(dstName,write_lst=False,spoof_nonessential=True,run_check=False)
