#Tiffany Miller and Jordan Hoppenheim

from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt

"""
Original file is located at
    https://colab.research.google.com/drive/1aXHxvy00I3OzWUz7qYytXJ3E9JM7FgY1
    
Generating, playing, plotting, analyzing a simple sound
    
The following assumes that a sound file "piano.wav" is saved on the Google Drive, in the "Colab Notebooks" directory.
"""

from google.colab import drive
drive.mount('/content/drive')

wav_piano = "/content/drive/My Drive/piano.wav"
Audio(wav_piano)

samplerate, data = wavfile.read(wav_piano)
print(f"sample rate = {samplerate}")
print(f"number of channels = 1")
n_datapoints = data.shape[0]
print(f"{n_datapoints} data points")
length_sec = n_datapoints / samplerate
print("length = {:2.3f} seconds".format(length_sec))

time = np.linspace(0., length_sec, data.shape[0])
plt.plot(time, data[:], label="mono channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

print(data[:])

plt.plot(data[int(0.2*n_datapoints):int(0.21*n_datapoints)], label="Left channel")
plt.show()

yf = np.fft.fft(data[:])
N = signal.shape[0]

fig, ax = plt.subplots()
ax.plot(np.abs(yf[:N//2]))
plt.show()

yf = np.fft.fft(data[:])
N = signal.shape[0]

fig, ax = plt.subplots()
plt.xlim([465, 500])
ax.plot(np.abs(yf[:N//2]))
plt.show()

yf = np.fft.fft(data[:])
N = signal.shape[0]

fig, ax = plt.subplots()
plt.xlim([940, 960])
ax.plot(np.abs(yf[:N//2]))
plt.show()

yf = np.fft.fft(data[:])
N = signal.shape[0]

fig, ax = plt.subplots()
plt.xlim([2380, 2395])
ax.plot(np.abs(yf[:N//2]))
plt.show()

yf = np.fft.fft(data[:])
N = signal.shape[0]

fig, ax = plt.subplots()
plt.xlim([1410, 1430])
ax.plot(np.abs(yf[:N//2]))
plt.show()

yf = np.fft.fft(data[:])
N = signal.shape[0]

fig, ax = plt.subplots()
plt.xlim([1890, 1910])
ax.plot(np.abs(yf[:N//2]))
plt.show()

"""Now we convert all the 7 peaks from frequency to time domaine. Then we combine all time domain component frequencies to get the orginal Piano"""

import numpy as np
f1 = 470.0              # in Hertz
f2 = 942.0
f3 = 2388.0
f4 = 1417.8
f5 = 1898.0           # in Hertz
sample_rate = 44100      # in Hertz
duration = 1            # in seconds
times = np.linspace(0, duration, duration*sample_rate)
signal1 = np.sin(2*np.pi*f1*times)

fig, ax = plt.subplots()
ax.plot(signal1[:200])
Audio(data=signal1, rate=sample_rate)

signal2 = 0.2*np.sin(2*np.pi*f2*times)
fig, ax = plt.subplots()
ax.plot(signal2[:200])
Audio(data=signal2, rate=sample_rate)

signal3 = 0.17*np.sin(2*np.pi*f3*times)
fig, ax = plt.subplots()
ax.plot(signal3[:200])
Audio(data=signal3, rate=sample_rate)

signal4 = 0.1*np.sin(2*np.pi*f4*times)
fig, ax = plt.subplots()
ax.plot(signal4[:200])
Audio(data=signal2, rate=sample_rate)

signal5 = 0.05*np.sin(2*np.pi*f5*times)
fig, ax = plt.subplots()
ax.plot(signal5[:200])
Audio(data=signal5, rate=sample_rate)

signal = signal1 + signal2 + signal3 + signal4 + signal5
fig, ax = plt.subplots()
ax.plot(signal[:1000])
Audio(data=signal, rate=sample_rate)

"""### Some simple transforms that should be understood / verified by hand calculation"""

np.fft.fft(np.array([1,0,0,0]))

np.fft.fft(np.array([0,1,0,0]))

np.fft.fft(np.array([0,0,1,0]))

np.fft.fft(np.array([0,0,0,1]))

np.fft.ifft(np.array([1,0,0,0]))

np.fft.ifft(np.array([0,1,0,0]))

np.fft.ifft(np.array([0,0,1,0]))

np.fft.ifft(np.array([0,0,0,1]))

trans = np.fft.fft(np.array([1,2,3,4]))
trans

np.fft.ifft(trans)

