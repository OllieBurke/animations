import numpy as np
import matplotlib.pyplot as plt

from pastamarkers import markers

f0 = 20        # Frequency
dt = 0.5/(2*f0)  # Shannon's sampling theorem, set dt < 1/2*highest_freq
T = 1000        # Final time
t = np.arange(0,T,dt)  # Time array


A = 1e-3 # Arbitrary amplitude
y = A * np.sin(2*np.pi*f0*t) # Signal waveform we wish to test

N_t = len(y)
freq = np.fft.fftshift(np.fft.fftfreq(len(y),dt)) # Frequencies

freq[N_t//2] = freq[N_t//2 + 1]
breakpoint()
df = abs(freq[1] - freq[0])  # Sample spacing in frequency

y_fft = dt * np.fft.fftshift(np.fft.fft(y)) # continuous time fourier transform [seconds]

# Plot results
plt.plot(freq,abs(y_fft)**2,  marker=markers.farfalle , ms = 20)
plt.xlabel(r'Frequency [Hz]',fontsize = 15)
plt.ylabel(r'$|\tilde{h}(f)|^2$',fontsize = 15)
plt.title(r'Periodigram',fontsize = 15)
plt.show()

# np.random.seed(1234)
# x = np.random.uniform(0,1,10)
# y = np.random.uniform(0,1,10)

# plt.scatter(x, y, marker=markers.mezzelune, s=5000, linewidth=0.5)
# plt.show()
quit()

PSD = 1e-6 * np.ones(N_t) # White noise, set PSD = constant. 

# Compute the SNRs
# Frequency domain SNR, squared CTFT
SNR2_f = 2 * np.sum(abs(y_fft)**2 / PSD) * df
# Time domain SNR, squared. Parseval's theorem applied.
SNR2_t = 2 * dt * np.sum(abs(y)**2 / PSD) 
# Analytical result, pen and paper. 
SNR2_t_analytical = (A**2) * T/PSD[0]

# Output
print("SNR squared in the frequency domain is =", SNR2_f)
print("SNR squared in the time domain is =", SNR2_t)
print("(pen and paper) Analytical result would predict SNR = ", SNR2_t_analytical)