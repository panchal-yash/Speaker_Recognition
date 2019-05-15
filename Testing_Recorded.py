import os
import _pickle as pickle
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import Record_audio_Fixed_Interval as recor 
import pandas as pd
import librosa
import logmmse
import time
import sys
sys.path.append(os.path.abspath("C:/Users/panch/Desktop/Version3/libs/filters/"))
import silence as silence_reduction
import Functions as FE


def Testing(user):
    source   = "Dataset/"   
    modelpath = "Models/"

    i=1
    ii=2

    user = str(user)

    fs, sig = wav.read(source+user+"/"+"Testing/"+user+" ("+str(i)+").wav" )
    Feature_DF = FE.Feature_Extraction(fs,sig,i,user)

    File_array = []
    Score_array = []
    Max_Array =["Max_Score"]

    path = "Models/"
    dirs = os.listdir(path)
    Testing_dir = source+user+"/"+"Testing/"

    for fileas in dirs:

        File_array.append(fileas) 
        #print(fileas)
        trained_model = open(path+fileas,"rb")
        gmm_Model = pickle.load(trained_model)
        score = gmm_Model.score(Feature_DF)
        Score_array = np.append(Score_array,score)
        
    File_array = pd.Series(File_array)    
    Score_array = pd.Series(Score_array)

    Max_value = max(Score_array)
    Max_Array.append(Max_value)

    Df = pd.DataFrame([File_array,Score_array])

    i=i+1

    while i<ii:

        fs, sig = wav.read(source+user+"/"+"Testing/"+user+" ("+str(i)+").wav" )
        Feature_DF = FE.Feature_Extraction(fs,sig,i,user)

        Score_array = []

        for fileas in dirs:

            trained_model = open(path+fileas,"rb")
            gmm_Model = pickle.load(trained_model)
            score = gmm_Model.score(Feature_DF)
            Score_array = np.append(Score_array,score)

        Score_array = pd.Series(Score_array)

        Max_value = max(Score_array)
        Max_Array.append(Max_value)
        
        Df=Df.append(Score_array,ignore_index=True)
        
        i=i+1

    mdf=pd.DataFrame(Max_Array)
    Df = pd.concat([Df, mdf], axis=1)
    Df.columns = Df.iloc[0]
    Df = Df[1:]

    Max = Df["Max_Score"]
    Current = Df[str(user)+".gmm"]
    Curt = str(user)+".gmm"

    if Df.iloc[0]["Max_Score"]==Df.iloc[0]["%s"%Curt]:
        x  = 'User is Authenticated!!!'
    else:
        x = 'User is Different'
    
    new_array = np.intersect1d(Max,Current)
    Correct=len(new_array)
    #print(Correct)
    accuracy = (Correct/(ii-1))*100
    #print(user,accuracy)
    Df.to_csv("Results/"+user+".csv")
    return x