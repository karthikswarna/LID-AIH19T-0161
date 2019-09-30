'''
s image will be saved in the current working directory which is to
be fed to our network
'''
import librosa
import numpy as np


def dataprepocess(imgSrc):

    expDur = 4
    sampling = 16000
    samples = sampling*expDur
    audio,sampling = librosa.load(imgSrc,sr=sampling)
    audio=audio[-sampling*5:]
    noFrames = 0
    #print(len(audio)%samples)
    if len(audio)>samples:
        noSec=np.floor(len(audio)/sampling)
        noFrames=noSec-expDur+1
    else:
        return None
    frames = []
    for i in range(int(noFrames)):
        melspec = librosa.feature.melspectrogram(
                audio[i*sampling:(i+expDur)*sampling], sr = sampling, n_mels = 129,
                fmax = 5000, n_fft = 1600, hop_length = 128)
        
        S_filter = librosa.decompose.nn_filter(melspec,aggregate=np.median,metric='cosine',width=int(librosa.time_to_frames(2, sr=sampling)))
        S_filter = np.minimum(melspec, S_filter)
        margin_i, margin_v = 2, 10
        power = 2
        mask_v = librosa.util.softmask(melspec - S_filter,margin_v * S_filter,power=power)
        melspec = mask_v * melspec
        melspec =  librosa.power_to_db(melspec,ref=np.max)
        melspec-=np.min(melspec)
        #print(np.max(clip),np.min(clip),np.max(melspec),np.min(melspec))
        melspec = melspec / np.max(melspec) - 0.5
        frames.append(melspec)

    framesArray = np.array(frames)
    framesArray = framesArray.reshape(int(noFrames),1,129,501)
    return framesArray


def datapreprocess_off(imgSrc):

    expDur = 4
    sampling = 16000
    samples = sampling*expDur
    audio,sampling = librosa.load(imgSrc,sr=sampling)
    
    noFrames = 0
    #print(len(audio)%samples)
    if len(audio)>samples:
        noSec=np.floor(len(audio)/sampling)
        noFrames=noSec-expDur+1
    else:
        return None
    frames = []
    for i in range(int(noFrames)):
        melspec = librosa.feature.melspectrogram(
                audio[i*sampling:(i+expDur)*sampling], sr = sampling, n_mels = 129,
                fmax = 5000, n_fft = 1600, hop_length = 128)

        S_filter = librosa.decompose.nn_filter(melspec,aggregate=np.median,metric='cosine',width=int(librosa.time_to_frames(2, sr=sampling)))
        S_filter = np.minimum(melspec, S_filter)
        margin_i, margin_v = 2, 10
        power = 2
        mask_v = librosa.util.softmask(melspec - S_filter,margin_v * S_filter,power=power)
        melspec = mask_v * melspec
        melspec =  librosa.power_to_db(melspec,ref=np.max)
        melspec-=np.min(melspec)
        #print(np.max(clip),np.min(clip),np.max(melspec),np.min(melspec))
        melspec = melspec / np.max(melspec) - 0.5
        frames.append(melspec)

    framesArray = np.array(frames)
    framesArray = framesArray.reshape(int(noFrames),1,129,501)
    return framesArray

