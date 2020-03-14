import numpy as np
import librosa
import matplotlib.pyplot as plot

y, sr = librosa.load('./igreja.wav') # carrega o audio em numpy
v, srv = librosa.load('./voz.wav')


n = 2*(np.convolve(v, y*0.02)) #realiza convolução de dois vetores
librosa.output.write_wav('./voz-na-igreja.wav', n, srv)


plot.subplot(211)
plot.title('igreja')

plot.plot(y)
plot.xlabel('Sample')
plot.ylabel('Amplitude')

plot.subplot(212)
plot.specgram(y, Fs=sr)
plot.xlabel('Time')
plot.ylabel('Frequency')

plot.show()
#-------------------------

plot.subplot(211)
plot.title('voz')

plot.plot(v)
plot.xlabel('Sample')
plot.ylabel('Amplitude')

plot.subplot(212)
plot.specgram(v, Fs= srv)
plot.xlabel('Time')
plot.ylabel('Frequency')


plot.show()

#-------------------------

plot.subplot(211)
plot.title('conv igreja-voz')

plot.plot(n)
plot.xlabel('Sample')
plot.ylabel('Amplitude')

plot.subplot(212)
plot.specgram(n, Fs= srv)
plot.xlabel('Time')
plot.ylabel('Frequency')


plot.show()