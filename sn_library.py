# Copyright (C) 2020  Marek Szczepanczyk <marek.szczepanczyk@ligo.org>
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


# 
# Date: Aug 26, 2020
# 


import numpy as np
from scipy import interpolate, fftpack, signal
from scipy.signal import butter, lfilter, freqz

# Constants

cm2kpc = 3.24078e-22
kpc2m  = 3.08567758128e+19 # m
D10kpc = 10.0 * kpc2m # 10 kpc in m

c = 2.99792458e8 # m/s
G = 6.67430e-11 # m^3 kg^−1 s^−2
msun = 1.9885e+30 # kg
esun = msun*c**2 # J


"""

The SN toolbox content:

>>> Preparation of the waveforms and the quadrupole moments
t,h   = sn_resample_wave(t,h,fs)
t,qij = sn_resample_quad(t,qij,fs)
x     = sn_remove_edges_wave(x,  left,right)
qij   = sn_remove_edges_quad(qij,left,right)
x     = sn_medfilt_wave(x,nr)
Qij   = sn_medfilt_quad(Qij,nr)
qij   = sn_wave2quad(h)
hp,hc = sn_create_waveform(qij,phi,theta)

>>> Analysis of waveform properties
f,H   = sn_fft_wave(t,h)
f,Qij = sn_fft_quad(t,qij)
hrss  = sn_hrss_wave(t,hp,hc)
hrss  = sn_hrss_quad(t,qij)
egw   = sn_egw_wave(t,hp,hc)
egw   = sn_egw_quad(t,qij)
egw   = sn_egw_quad_freq(t,qij,D)
f, dEdf  = sn_dedf_quad_freq(t,qij,D)
egw_evol = sn_egw_evolution_wave(t,hp,hc,D)
egw_evol = sn_egw_evolution_quad(t,qij,D)
f,hchar = sn_hchar_wave(t,h)
f,hchar = sn_hchar_quad(t,qij)
fpeak,hpeak = sn_fpeak_wave(f,h)
fpeak,hpeak = sn_fpeak_quad(f,qij)
snr   = sn_snr_wave(t,h,fasd,asd)
snr   = sn_snr_quad(t,qij,fasd,asd)

"""


def sn_resample_wave(t,h,fs):
    """
    Interpolate array h to the fs sampling frequency.
    
    Input:
        t  - time array, in seconds
        h  - strain array to be interpolated
        fs - sampling frequency
    Output:
        t1 - time array, after resampling
        h1 - new strain array
    """
    
    # Quick check
    if len(t)!=len(h):
        print("Error: t and h need to have equal sizes")
        return 0
    
    # Define new time with fs
    t1 = np.arange(t[0],t[-1],1.0/fs)
    
    # Interpolation
    tck = interpolate.splrep(t,h,s=0)
    h1  = interpolate.splev(t1,tck,der=0)
    
    return t1, h1


def sn_resample_quad(t,qij,fs):
    """
    Interpolate each component of quadrupole moment to the fs sampling frequency.
    
    Input:
        t   - time array, in seconds
        qij - quadrupole moment N x 3 x 3 array to be interpolated
        fs  - sampling frequency
    Output:
        t1   - time array, after resampling
        qij1 - new quadrupole moment array
    """

    # Find new time array
    t1,tmp = sn_resample_wave(t,qij[:,0,0],fs)

    # Define new quadrupole moment array
    qij1 = np.zeros((len(t1),3,3),dtype=float)

    # Interpolate each quadrupole moment
    t1, qij1[:,0,0] = sn_resample_wave(t, qij[:,0,0], fs)
    t1, qij1[:,0,1] = sn_resample_wave(t, qij[:,0,1], fs)
    t1, qij1[:,0,2] = sn_resample_wave(t, qij[:,0,2], fs)
    t1, qij1[:,1,1] = sn_resample_wave(t, qij[:,1,1], fs)
    t1, qij1[:,1,2] = sn_resample_wave(t, qij[:,1,2], fs)
    t1, qij1[:,2,2] = sn_resample_wave(t, qij[:,2,2], fs)
    qij1[:,1,0] = qij1[:,0,1]
    qij1[:,2,0] = qij1[:,0,2]
    qij1[:,2,1] = qij1[:,1,2]    

    return t1, qij1


def sn_remove_edges_wave(x,left,right):
    """
    Remove edges from the time series e.g.
    >>> t = sn_remove_edges(t,0.035,0)
    It removes 3.5% of data points at the beginning of time series
    
    Input:
        x - input array (time, strain etc.)
    Output:
        x - output array
    """
    
    # Calculate how many data points need to be removed from the begining and end
    n = len(x)
    nleft = int(n*left)
    nright = int(n*right)
    
    # Crop input array
    x = x[nleft:n-nright]
    
    return x


def sn_remove_edges_quad(qij,left,right):
    """
    Remove edges from the time series e.g.
    >>> qij = sn_remove_edges(qij,0.035,0)
    It removes 3.5% of data points at the beginning of each quadrupole moment components
    
    Input:
        qij  - input quadrupole moment, array N x 3 x 3
    Output:
        qij1 - output quadrupole moment
    """

    # Find length of the croped array
    tmp = sn_remove_edges_wave(qij[:,0,0],left,right)
    N = (len(tmp))

    # Define new quadrupole moment array
    qij1 = np.zeros((N,3,3),float)
    #qij  = np.zeros((len(h),3,3),float)

    # Crop each element of the quadrupole moment
    qij1[:,0,0] = sn_remove_edges_wave(qij[:,0,0],left,right)
    qij1[:,0,1] = sn_remove_edges_wave(qij[:,0,1],left,right)
    qij1[:,0,2] = sn_remove_edges_wave(qij[:,0,2],left,right)
    qij1[:,1,1] = sn_remove_edges_wave(qij[:,1,1],left,right)
    qij1[:,1,2] = sn_remove_edges_wave(qij[:,1,2],left,right)
    qij1[:,2,2] = sn_remove_edges_wave(qij[:,2,2],left,right)
    qij1[:,1,0] = qij1[:,0,1]
    qij1[:,2,0] = qij1[:,0,2]
    qij1[:,2,1] = qij1[:,1,2]    

    return qij1


def sn_medfilt_wave(x,nr):
    """
    Average data point, primarily for plotting of FFT or spectra
    
    Input:
        x  - input array
        nr - number of points to average
    Output:
        x  - output array of same size as the input
    """

    x = signal.medfilt(abs(x),nr)

    return x


def sn_medfilt_quad(Qij,nr):
    """
    Average data points of 3 x 3 quadrupole matrix, primarily for plotting of FFT or spectra
    
    Input:
        Qij - input N x 3 x 3 quadrupole moment array
        nr  - number of points to average
    Output:
        Qij - Output N x 3 x 3 quadrupole moment array
    """

    Qij[:,0,0] = signal.medfilt(abs(Qij[:,0,0]),nr)
    Qij[:,0,1] = signal.medfilt(abs(Qij[:,0,1]),nr)
    Qij[:,0,2] = signal.medfilt(abs(Qij[:,0,2]),nr)
    Qij[:,1,1] = signal.medfilt(abs(Qij[:,1,1]),nr)
    Qij[:,1,2] = signal.medfilt(abs(Qij[:,1,2]),nr)
    Qij[:,2,2] = signal.medfilt(abs(Qij[:,2,2]),nr)


    return Qij


def sn_wave2quad(h):
    """
    Build quadrupole moment from a wave h, specifically for 2D simulations.
    The quadrupole moments need to be traceless, see:
    
    Finn+90: https://ui.adsabs.harvard.edu/abs/1990ApJ...351..588F/abstract    
    
    Input:
        h - strain array, length N
    Output:
        qij - N x 3 x 3 output quadrupole array
    """
    
    qij = np.zeros((len(h),3,3),float)
    qij[:,0,0] = -1.0/3.0 * h
    qij[:,1,1] = -1.0/3.0 * h
    qij[:,2,2] =  2.0/3.0 * h

    return qij


def sn_create_waveform(qij,phi,theta):
    """
    Create a waveform from quadrupole moment.
    Equations:
    
    2D case:
        Finn+90, eqn (32)
        https://ui.adsabs.harvard.edu/abs/1990ApJ...351..588F/abstract
        
    3D case:  
        OOhara+07, eqn (2.10)
        https://academic.oup.com/ptps/article/doi/10.1143/PTPS.128.183/1930275
    
        Andresen+17, eqn (7-10)
        https://academic.oup.com/mnras/article/468/2/2032/3070429
    
    Input:
        phi, theta - supernova source coordinates (spherical)
        qij - quadrupole moment
    Output:
        hp, hc - plus and cross polarization components
    """
    
    qxx = qij[:,0,0]
    qxy = qij[:,0,1]
    qxz = qij[:,0,2]
    qyy = qij[:,1,1]
    qyz = qij[:,1,2]
    qzz = qij[:,2,2]
    qzy = qyz
    qzx = qxz
    qyx = qxy
    
    N = len(qxx)
    hp = np.zeros(N)
    hc = np.zeros(N)
    
    # Is simulation 2D or 3D?
    qtmp = qxy+qxz+qyz
    
    if sum(abs(qtmp))<1.0e-10*sum(abs(qzz)): # 1.0e-10 factor is arbitrary
        # 2D simulation
        #print("2D simulation")
        hp = 3.0/2.0 * np.sin(theta)**2 * qzz
        
    else:
        # 3D simulation:
        qthetatheta = np.zeros(N)
        qphiphi     = np.zeros(N)
        qthetaphi   = np.zeros(N)

        qthetatheta = (qxx*np.cos(phi)**2 + qyy*np.sin(phi)**2+2*qxy*np.sin(phi)*np.cos(phi))*np.cos(theta)**2+\
                      qzz*np.sin(theta)**2 - 2*(qxz*np.cos(phi)+qyz*np.sin(phi))*np.sin(theta)*np.cos(theta)
        qphiphi     = qxx*np.sin(phi)**2 + qyy*np.cos(phi)**2 - 2*qxy*np.sin(phi)*np.cos(phi)
        qthetaphi   = (qyy-qxx)*np.cos(theta)*np.sin(phi)*np.cos(phi)+ \
                     qxy*np.cos(theta)*(np.cos(phi)**2-np.sin(phi)**2) + \
                     qxz*np.sin(theta)*np.sin(phi) - qyz*np.sin(theta)*np.cos(phi)
        hp = qthetatheta - qphiphi 
        hc = 2.0*qthetaphi
    
    return hp,hc



def sn_fft_wave(t,h):
    """
    Fourier transform of the h array.
    Time array need to be evenly spaced
    
    Input:
        t - time array
        h - strain array
    Output:
        f - frequency array
        H - one-sided FFT with doubled amplitudes
    """
    dt = t[1] - t[0]
    
    # Fourier Transform
    H = dt*fftpack.fft(h)
    f = fftpack.fftfreq(len(t),dt)
    
    # shift frequencies and data into normal order
    f = fftpack.fftshift(f)
    H = fftpack.fftshift(H)
    
    # use only real data from n/2+1 (freq>0) to n-1
    # double- to one-sided
    n = len(f)
    f =     f[int(n/2)+1:n-1]
    H = 2.0*H[int(n/2)+1:n-1]
    
    return f, H


def sn_fft_quad(t,qij):
    """
    FFT of each quadrupole moment components
    
    Input:
        t   - time array
        qij - quadrupole momoent components
    Output:
        f   - frequency array
        Qij - FFT of each quadrupole moment components
    
    """

    f, tmp = sn_fft_wave(t,qij[:,0,0])
    Qij = np.zeros((len(f),3,3),dtype=complex)
    Qij[:,0,0] = tmp
    f, Qij[:,0,1] = sn_fft_wave(t,qij[:,0,1])
    f, Qij[:,0,2] = sn_fft_wave(t,qij[:,0,2])
    f, Qij[:,1,1] = sn_fft_wave(t,qij[:,1,1])
    f, Qij[:,1,2] = sn_fft_wave(t,qij[:,1,2])
    f, Qij[:,2,2] = sn_fft_wave(t,qij[:,2,2])
    Qij[:,1,0] = Qij[:,0,1]
    Qij[:,2,0] = Qij[:,0,2]
    Qij[:,2,1] = Qij[:,1,2]

    return f,Qij


def sn_hrss_wave(t,hp,hc):
    """
    Calculate hrss value for a strain:
    hrss = sqrt[int( ( hp^2 + hc^2)*dt ) ]
    
    Input:
        t  - time array
        hp - strain array
    Output:
        hrss - hrss value
    """
    
    h  = hp**2 + hc**2
    h  = h[0:-1] # remove last element to match size of dt and perform integration
    dt = np.diff(t)
    
    hrss = np.sqrt( sum(h*dt) )
    
    return hrss


def sn_hrss_quad(t,qij,D):
    """
    Calculate angle averaged hrss using quadrupole moment.
    Factor (D * c**4 / G) gives physical meaning for the quadrupole moments.
    
    hrss = sqrt[ average( int( ( hp^2 + hc^2)*dt ) ) ]
    where hp and hc are expressed in terms quadrupole moment

    Input:
        t   - time array
        qij - quadrupole moment
    Output:
        hrss - source angle averaged hrss
    """    
    
    dt = t[1]-t[0]
    
    hrss_sq = (D * c**4 / G)**2 * 8.0/15.0 * (qij[:,0,0]**2 + qij[:,1,1]**2 + qij[:,2,2]**2 + \
            3.0*(qij[:,0,1]**2 + qij[:,0,2]**2 + qij[:,1,2]**2) - \
            qij[:,0,0]*qij[:,1,1] - qij[:,0,0]*qij[:,2,2] - qij[:,1,1]*qij[:,2,2])

    hrss =  G/(c**4  * D) * np.sqrt(sum(hrss_sq * dt))
    
    return hrss



def sn_egw_wave(t,hp,hc,D):
    """
    Calculate GW energy given hp and hc, see:
    a) Maggiore Vol 1, eqn (1.156)
    b) Mueller+12, eqn (63) https://ui.adsabs.harvard.edu/abs/2012A%26A...537A..63M/abstract
    
    Input:
        t  - time
        hp - plus polarization
        hc - cross polarization
        D  - source distance in meters
    Output:
        egw - GW energy
    """
    
    dt  = t[2]-t[1]
    dhp_dt = np.diff(hp) / np.diff(t)
    dhc_dt = np.diff(hc) / np.diff(t)
    
    egw = c**3/(4.0*G) * D**2 * sum( ((dhp_dt)**2 + (dhc_dt)**2)*dt )
    
    return egw



def sn_egw_quad(t,qij,D):
    """
    GW energy calculation in time domain using quadrupole moment.
    Factor (D * c**4 / G) gives physical meaning for the quadrupole moments.
    Notice: G / (5.0*c**5) * (D * c**4 / G )**2 = c**3 / (4*G) * D**2
    See Szczepanczyk+15: https://dcc.ligo.org/LIGO-T1500586
    
    Input:
        t   - time array
        qij - quadrupole moment
        D   - source distance in meters
    Output:
        egw - GW energy
    """
    
    dt  = t[2]-t[1]

    dqij_dt = np.zeros((len(t)-1,3,3))
    
    trace = qij[:,0,0] + qij[:,1,1] + qij[:,2,2]
    qij[:,0,0] = qij[:,0,0] - 1.0/3*trace
    qij[:,1,1] = qij[:,1,1] - 1.0/3*trace
    qij[:,2,2] = qij[:,2,2] - 1.0/3*trace
    dqij_dt[:,0,0] = np.diff(qij[:,0,0]) / np.diff(t)
    dqij_dt[:,0,1] = np.diff(qij[:,0,1]) / np.diff(t)
    dqij_dt[:,0,2] = np.diff(qij[:,0,2]) / np.diff(t)
    dqij_dt[:,1,1] = np.diff(qij[:,1,1]) / np.diff(t)
    dqij_dt[:,1,2] = np.diff(qij[:,1,2]) / np.diff(t)
    dqij_dt[:,2,2] = np.diff(qij[:,2,2]) / np.diff(t)
    
    
    dEdt = G / (5.0*c**5) * (D * c**4 / G )**2 * \
            (dqij_dt[:,0,0]**2 + dqij_dt[:,1,1]**2 + dqij_dt[:,2,2]**2 + \
            2*(dqij_dt[:,0,1]**2 + dqij_dt[:,0,2]**2 + dqij_dt[:,1,2]**2))

    egw = sum( dEdt * dt )
    
    return egw


def sn_egw_quad_freq(t,qij,D):
    """
    GW energy calculation in frequency domain using quadrupole moment.
    Factor (D * c**4 / G) gives physical meaning for the quadrupole moments.
    For details see (eqn (9)): https://dcc.ligo.org/LIGO-T1500586
    
    Input:
        t   - time array
        qij - quadrupole moment
        D   - source distance in meters
    Output:
        egw - GW energy
    """
    
    f,Qij = sn_fft_quad(t,qij)
    
    df = f[2]-f[1]
    trace = Qij[:,0,0] + Qij[:,1,1] + Qij[:,2,2]
    dEdf = G / (5.0*c**5) * (D * c**4 / G )**2 *(2.0*np.pi*f)**2 * \
            (abs(Qij[:,0,0]-1.0/3*trace)**2 + \
             abs(Qij[:,1,1]-1.0/3*trace)**2 + \
             abs(Qij[:,2,2]-1.0/3*trace)**2 + \
             2.0*(abs(Qij[:,0,1])**2 + abs(Qij[:,1,2])**2 + abs(Qij[:,2,0])**2))
    
    egw_freq = sum( dEdf * df )
    
    #sn_fft_quad calculates one-sided fft with doubled amplitudes.
    #To use eqn from T1500586 correclty, egw needs to be divided by 2:
    egw_freq = egw_freq / 2
    
    return egw


def sn_dedf_quad_freq(t,qij,D):
    """
    GW energy spectrum
    
    Input:
        t   - time array
        qij - quadrupole moment
        D   - source distance in meters
    Output:
        dEdf - dE / df
    """
    
    f,Qij = sn_fft_quad(t,qij)
    
    df = f[2]-f[1]
    trace = Qij[:,0,0] + Qij[:,1,1] + Qij[:,2,2]
    dEdf = G / (5.0*c**5) * (D * c**4 / G )**2 *(2.0*np.pi*f)**2 * \
            (abs(Qij[:,0,0]-1.0/3*trace)**2 + \
             abs(Qij[:,1,1]-1.0/3*trace)**2 + \
             abs(Qij[:,2,2]-1.0/3*trace)**2 + \
             2.0*(abs(Qij[:,0,1])**2 + abs(Qij[:,1,2])**2 + abs(Qij[:,2,0])**2))

    #sn_fft_quad calculates one-sided fft with doubled amplitudes.
    #To use eqn from T1500586 correclty, egw needs to be divided by 2:
    dEdf = dEdf / 2
    
    return f, dEdf


def sn_egw_evolution_wave(t,hp,hc,D):
    """
    Calculate GW energy given hp and hc, see Maggiore Vol 1, eqn (1.156)
    
    Input:
        t  - time array
        hp - plus polarization array
        hc - cross polarization array
        D  - source distance in meters
    Output:
        egw_evol - cummulative GW energy evolution array
    """
    
    dt  = t[2]-t[1]
    dhp_dt = np.diff(hp) / np.diff(t)
    dhc_dt = np.diff(hc) / np.diff(t)
    
    egw_array = c**3/(4.0*G) * D**2 * ((dhp_dt)**2 + (dhc_dt)**2)*dt 

    egw_evol = np.cumsum(egw_array)

    # Add zero element at the beginning to match the size of array t
    egw_evol = np.insert(egw_evol,0,0)

    return egw_evol


def sn_egw_evolution_quad(t,qij,D):
    """
    Calculate GW energy evolution using quadrupole moment.
    Factor (D * c**4 / G) gives physical meaning for the quadrupole moments
    
    Input:
        t   - time array
        qij - quadrupole moment array
        D   - source distance in meters
    Output:
        egw_evol - cummulative GW energy evolution array
    """
        
    dt  = t[2]-t[1]
    

    dqij_dt = np.zeros((len(t)-1,3,3))

    dqij_dt[:,0,0] = np.diff(qij[:,0,0]) / np.diff(t)
    dqij_dt[:,0,1] = np.diff(qij[:,0,1]) / np.diff(t)
    dqij_dt[:,0,2] = np.diff(qij[:,0,2]) / np.diff(t)
    dqij_dt[:,1,1] = np.diff(qij[:,1,1]) / np.diff(t)
    dqij_dt[:,1,2] = np.diff(qij[:,1,2]) / np.diff(t)
    dqij_dt[:,2,2] = np.diff(qij[:,2,2]) / np.diff(t)


    dEdt = G / (5.0*c**5) * (D * c**4 / G )**2 * \
            (dqij_dt[:,0,0]**2 + dqij_dt[:,1,1]**2 + dqij_dt[:,2,2]**2 + \
            2*(dqij_dt[:,0,1]**2 + dqij_dt[:,0,2]**2 + dqij_dt[:,1,2]**2))

    egw_array = dEdt*dt

    egw_evol = np.cumsum(egw_array)

    # Add zero element at the beginning to match the size of array t
    egw_evol = np.insert(egw_evol,0,0)

    return egw_evol



def sn_hchar_wave(t,h):
    """
    Calculating of a characteristic strain, hchar, for a given waveform.
    Characteristic strain is dimensionless and it is defined as:
    hchar = f * fft(h)
    See for example Moore+14: https://arxiv.org/pdf/1408.0740.pdf
    
    Input:
        t - time array
        h - strain array
    Output:
        f - frquency array
        hchar - charasteristic strain array
    """
    
    f, H  = sn_fft_wave(t,h)
    hchar = f * H
    
    hchar = abs(hchar)

    return f, hchar


def sn_hchar_quad(t,qij):
    """
    Calculating characteristic strain, hchar, for a given quadrupole moment.
    See:
    Fladanan&Hughes+98 (eqn 5.2): https://journals.aps.org/prd/abstract/10.1103/PhysRevD.57.4535
    Yakunin+15 (eqn 7): https://arxiv.org/abs/1505.05824
    Szczepanczyk+15 (eqn 9): https://dcc.ligo.org/LIGO-T1500586
    and sn_egw_quad function
    
    Input:
        t   - time array
        qij - quadrupole moment array
    Output:
        f - frequency array (f>0)
        hchar - characteristic strain
    """

    f,Qij = sn_fft_quad(t,qij)

    trace = Qij[:,0,0] + Qij[:,1,1] + Qij[:,2,2]
    dEdf = G / (5.0*c**5) * ( c**4 / G )**2 *(2.0*np.pi*f)**2 * \
            (abs(Qij[:,0,0]-1.0/3*trace)**2 + \
             abs(Qij[:,1,1]-1.0/3*trace)**2 + \
             abs(Qij[:,2,2]-1.0/3*trace)**2 + \
             2.0*(abs(Qij[:,0,1])**2 + abs(Qij[:,1,2])**2 + abs(Qij[:,2,0])**2))

    #sn_fft_quad calculates one-sided fft with doubled amplitudes.
    #To use eqn from T1500586 correclty, egw needs to be divided by 2:
    dEdf = dEdf / 2.0
    
    hchar = np.sqrt( 2.0 * G / (np.pi**2 * c**3 ) * dEdf )
    
    return f, hchar



def sn_fpeak(f,spect):
    """
    Calculate a peak frequency given characteristic strain.
    
    Input:
        t - time array
        h - strain array
    Output:
        fpeak - peak frquency
        hpeak - value at the peak frquency
    """

    maxid = np.argmax(spect)
    fpeak = f[maxid]
    hpeak = spect[maxid]
    
    return fpeak, hpeak



def sn_snr_wave(t,h,fasd,asd):
    """
    Calculate signal-to-noise ratio for a given waveform and Amplitude Spectral Density (ASD)
    General equation for two-sided fft:
    
    SNR^2 = 4 * int_0^infty ( |h(f)|^2 / Sn(f) df)
    
    Below we drop factor of '4' because sn_fft() function already calculates one-sided fft.
    
    Input:
        t - time array
        h - strain array
        fasd - frequency array for noise ASD
        asd  - array of noise ASD
    Output:
        snr - signal-to-noise ratio
    """
    
    # Do fft of a waveform
    f1, FFT  = sn_fft_wave(t,h)    
    FFT = abs(FFT)
    
    # Adjust fft array to asd array
    tck = interpolate.splrep(fasd,asd,s=0)
    asd = interpolate.splev(f1,tck,der=0) 

    df = f1[2]-f1[1]
    psd = asd**2
    
    # Calculate SNR
    snrsq = sum(FFT**2/psd)*df
    snr = np.sqrt(snrsq)

    return snr



def sn_snr_quad(t,qij,fasd,asd):
    """
    Calculate signal-to-noise ratio for a given waveform and Amplitude Spectral Density (ASD)
    See Fladanan&Hughes+98 (eqn 5.2): https://journals.aps.org/prd/abstract/10.1103/PhysRevD.57.4535
    
    SNR^2 = 4 * int_0^infty ( |hchar(f)|^2 / hn(f) df/f)
    where:
    hn(f) = f * Sn(f)
    
    Below we drop factor of '4' because sn_fft() function already calculates one-sided fft 
    with doubled amplitudes.
    
    Input:
        t    - time array
        qij  - quadrupole moment
        fasd - frequency array for ASD
        asd  - ASD array
    Outpu:
        snr - signal-to-noise ratio
    """
    
    # Calculate characteristic strain
    fchar, hchar = sn_hchar_quad(t,qij)

    # Adjust fft array to asd array
    tck = interpolate.splrep(fasd,asd,s=0)
    asd = interpolate.splev(fchar,tck,der=0) 

    psd = asd**2 

    df = fchar[2]-fchar[1]

    # Calculate SNR
    snrsq = sum(hchar**2/(fchar**2*psd))*df
    snr = np.sqrt(snrsq)
    
    return snr



