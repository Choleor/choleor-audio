# Extract Mean Amplitude
import librosa
import numpy as np

def getAmplitude(file):
    music, fs = librosa.load(file)
    music_stft = librosa.stft(music)
    music_amp = librosa.amplitude_to_db(abs(music_stft))
    music_fin_amp = [[0 for i in range(len(music_amp[j]))] for j in range(len(music_amp))]

    for i in range(len(music_amp)):
        for j in range(len(music_amp[i])):
            if (music_amp[i][j] >= 0):
                music_fin_amp[i][j] = music_amp[i][j]

    length = len(music_fin_amp)  # dB array 길이: 1025
    len_unit = int(length / 9)  # 8박자 단위로 자르기 때문에 9로 나누기
    amp_list = []

    for i in range(9):
        amp = round(np.mean(music_fin_amp[(len_unit * i):(len_unit * (i + 1))]), 2)
        if (i == 8):
            amp = round(np.mean(music_fin_amp[(len_unit * 8):length]), 2)

        if (amp < 0 or amp > 30):
            step = 0
        elif (amp >= 0 and amp <= 3):
            step = 1
        elif (amp > 3 and amp <= 6):
            step = 2
        elif (amp > 6 and amp <= 9):
            step = 3
        elif (amp > 9 and amp <= 12):
            step = 4
        elif (amp > 12 and amp <= 15):
            step = 5
        elif (amp > 15 and amp <= 18):
            step = 6
        elif (amp > 18 and amp <= 21):
            step = 7
        elif (amp > 21 and amp <= 24):
            step = 8
        elif (amp > 24 and amp <= 27):
            step = 9
        else:
            step = 10

        amp_list.append(step)  # 자른 박자 안에서 평균 진폭 구하기

    return amp_list