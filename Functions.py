import numpy as np
import os
import sys
sys.path.append(os.path.abspath("C:/Users/panch/Desktop/Version3/libs/filters/"))
import silence as silence_reduction 
import pandas as pd
import librosa
import logmmse
from python_speech_features import mfcc


def Feature_Extraction(fs,sig,user,i):

    audio= silence_reduction.remove_silence(fs,sig)
    audio = logmmse.logmmse(audio,fs)
    audio=sig
    file_name = str(user) + "("+str(i)+")"
    vector = mfcc(sig,fs)
    vector_df = pd.DataFrame(vector)
                
    mfcc_delta = librosa.feature.delta(vector)
    mfcc_delta_df = pd.DataFrame(mfcc_delta)

    mfcc_delta2 = librosa.feature.delta(vector, order=2)
    mfcc_delta2_df = pd.DataFrame(mfcc_delta2)
    
    Features = pd.concat([vector_df, mfcc_delta_df,mfcc_delta2_df], axis=1)
    FINAL_FEATURES = pd.DataFrame(Features)

    FINAL_FEATURES.set_axis(list(range(0, 39)), axis='columns', inplace=True)
    Feature_DF = pd.DataFrame(FINAL_FEATURES)

    return Feature_DF