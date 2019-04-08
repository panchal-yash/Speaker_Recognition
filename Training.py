#python2.7

#TRAINING MODE
import pickle
import json
import numpy as np
from sklearn.mixture import GaussianMixture as GMM 
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import pandas as pd
import Record_audio_Fixed_Interval as recorder_file
import librosa
import numpy as np
import logmmse
import sys
import scipy.io.wavfile as wavfile
import numpy as np
import os
sys.path.append(os.path.abspath("C:/Users/panch/Desktop/Version3/libs/filters/"))
import silence as silence_reduction
import Functions as FE


source   = "Dataset/"   
dest = "Speakers_models/"
user = input("Training Enter Location of file: Dataset/")
i=1
ii=4

Vector_Features = []
Features =  []
Series = pd.Series()

fs, sig = wav.read(source+user+"/"+"Testing/"+user+" ("+str(i)+").wav" )
Feature_DF = FE.Feature_Extraction(fs,sig,user,i)

i=i+1

while i < ii:
   
    fs, sig = wav.read(source+user+"/"+"Testing/"+user+" ("+str(i)+").wav" )
    FINAL_FEATURES = FE.Feature_Extraction(fs,sig,user,i)
    Feature_DF = pd.concat([Feature_DF, FINAL_FEATURES], ignore_index=True, sort=False)
    print(i)
    i =i+1

gmm = GMM(n_components = 13,max_iter = 1000, covariance_type='diag',n_init = 1)
gmm.fit(Feature_DF)
    
print("Model Trained")
picklefile = str(user)+".gmm"
pickle.dump(gmm,open('Models/' + picklefile,"wb"),protocol=2)