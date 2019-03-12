import numpy as np
import librosa
import speech_recognition as sr
import matplotlib.pyplot as plt
from os import path
from scipy.io import wavfile

def fourier(signal_in):
    fourier = np.fft.rfft(signal_in)
    fourier_meg = np.abs(fourier)
    return fourier_meg
    
def check_speaking(arr):
    if (np.count_nonzero(arr) != 0): #One of the number in the array are non-zero, that means someone is speaking
        return False
    return True   

def google_recongnition(audio_file):
    AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), audio_file)
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    #google recognition
    try:
        print(r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

signal, sample_rate = librosa.load("travel09.wav",sr = None, mono = True)
window_size = 5
thresholds = 0.18 #the noise threshold
weights = np.ones(window_size)
avg_signal = np.convolve(signal, weights, 'same')
abs_avg_signal = np.abs(avg_signal)
abs_avg_signal[abs_avg_signal<thresholds] = 0

#detecting if the person is speaking
speaking_signal = np.zeros(np.size(signal))
for index, value in enumerate(abs_avg_signal):
    if (value > 0):
        speaking_signal[index] = True

#noise reduce
sounds_after_noise_reduce = np.zeros(np.size(signal))
for index, value in enumerate(speaking_signal):
    if (value == True):
        sounds_after_noise_reduce[index] = signal[index]


#divide the singal to segments

time_stamp = 2300; start = 0; end = 0;  #change the time_stamp to find the silence when other person is talking
for index, value in enumerate(sounds_after_noise_reduce):
    if ((value != 0)and(start == 0)):
        start = index
    elif((start != 0) and (value == 0) and (check_speaking(sounds_after_noise_reduce[index:index+time_stamp])==True)): 
        end = index
        if ((end > start) and (start != 0)):
            seg = np.array(signal[start:end])
            start = 0; end = 0
            freq = np.argmax(fourier(seg))
            #print("freq: " + str(freq))
            #plt.figure()
            #plt.plot(seg)
            #plt.show()   uncomment these three lines to visualize the signal and freq, also you may want to access the audio file
            y = (np.iinfo(np.int32).max * (seg/np.abs(seg).max())).astype(np.int32) #convert double to int
            wavfile.write("output.wav", sample_rate, y)
            if (freq < 200):
                print("male: ",end ="" )
                google_recongnition("output.wav")
            else:
                print("female: ",end ="" )
                google_recongnition("output.wav")    
            