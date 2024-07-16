import sys
import os

import matplotlib.pyplot as plt
import numpy as np

from few.trajectory.inspiral import EMRIInspiral
from few.waveform import GenerateEMRIWaveform 

from scipy.io.wavfile import write
from scipy import signal
import matplotlib.colors as colors


use_gpu = False

# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs={
        "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
        "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    }

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(1e3),  # all of the trajectories will be well under len = 1000
    "use_gpu": use_gpu  # GPU is available in this class
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": False,
}

wave_generator = GenerateEMRIWaveform('Pn5AAKWaveform', 
            inspiral_kwargs = inspiral_kwargs, amplitude_kwargs = amplitude_kwargs,
            Ylm_kwargs = Ylm_kwargs, sum_kwargs = sum_kwargs)

MTSUN_SI = 4.925491025873693e-06

# Parameters
duration_seconds = 5
sample_rate = 44100  # Adjust as needed
frequency_Hz = 100.0  # Adjust to the desired frequency

# Generate time array
t = np.arange(0, duration_seconds, 1 / sample_rate)

# Generate a sinusoidal waveform
audio_signal = 0.5 * np.sin(2 * np.pi * frequency_Hz * t)

# Scale to 16-bit PCM audio range
scaled_signal = np.int16(audio_signal * 32767)

# Save the audio file
write('single_frequency_audio.wav', sample_rate, scaled_signal)

# EMRI parameters
# Initial conditions
M = 1e6
mu = 100.0

mass_ratio = mu/M

a = 0.9
p0 = 10.0
e0 =  0.5
iota0 = 0.3
Y0 = np.cos(iota0)

dist = 1
qS = 0.5
phiS = 0.5
qK = 0.5
phiK = 0.5

Phi_phi0 = 0
Phi_theta0 = 0
Phi_r0 = 0 

a0 = p0/(1-e0**2)
f0 = a0**(-3/2) / (M * MTSUN_SI * np.pi * 2)
theta, phi = np.pi/3, np.pi/3

# Set up length of trajectory in time (seconds)
T = 1e4 / (365*24*60*60)
dt = 10 

# base frequency
print("base frequency [Hz]:", f0, "desired frequency [Hz]:", frequency_Hz)
# new mass to obtain the signal in the desired frequency
newM = M * f0 / frequency_Hz


# specific_modes = [(ll,2,0) for ll in range(2,10)]
specific_modes = None

dt = 1/sample_rate
breakpoint()
print("Generating wave")
wave_base = wave_generator(newM, mass_ratio * newM, a, p0, e0, Y0, dist, qS, phiS, qK, phiK, 
                          Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, dt=dt, T=T) 
h = wave_base.real
print("Finished wave")

# Scale the signal to make it loud enough
scaled_signal = np.int16(h * 32767)

print("Writing to file")
# Save the audio file
write('emri_audio.wav',sample_rate, scaled_signal)