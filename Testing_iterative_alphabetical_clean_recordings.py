import os
import _pickle as pickle
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import Record_audio_Fixed_Interval as recor 
import pandas as pd
import librosa
import logmmse
#import warnings
#warnings.filterwarnings("ignore")
import time
import sys
sys.path.append(os.path.abspath("C:/Users/panch/Desktop/Version3/libs/filters/"))
sys.path.append(os.path.abspath("C:/Users/panch/Desktop/Version3/libs/feature/"))
import silence as silence_reduction


#path to training data
source   = "Dataset/"   

#path where training speakers will be saved
modelpath = "Models/"

for charact_no in range(97,111): 
    #user = input("Enter user you want to test: ")    
    #i = 1 
    user = chr(charact_no)
    i=1
    ii=5
    #ii=int(input("Enter No of Tests: "))



    fs, sig = wav.read(source+user+"/"+"Testing/"+user+" ("+str(i)+").wav" )
    #audio= silence_reduction.remove_silence(fs,sig)
    #audio = logmmse.logmmse(audio,fs)
    #sig=audio
    file_name = user+" ("+str(i)+")"

    vector = mfcc(sig,fs)
    vector_df = pd.DataFrame(vector)
    #vector = librosa.feature.mfcc(sig,fs)
            

        
    mfcc_delta = librosa.feature.delta(vector)
    mfcc_delta_df = pd.DataFrame(mfcc_delta)

    mfcc_delta2 = librosa.feature.delta(vector, order=2)
    mfcc_delta2_df = pd.DataFrame(mfcc_delta2)

    
    Features = pd.concat([vector_df, mfcc_delta_df,mfcc_delta2_df], axis=1)
    FINAL_FEATURES = pd.DataFrame(Features)

    FINAL_FEATURES.set_axis(list(range(0, 39)), axis='columns', inplace=True)
    
    Feature_DF = pd.DataFrame(FINAL_FEATURES)


    #vector = librosa.feature.mfcc(y=sig,sr=fs,n_mfcc=40)
    #vector_df = pd.DataFrame(vector)

    #vector = mfcc(audio,fs)
    #vector_df = pd.DataFrame(vector)
        
    #mfcc_delta = librosa.feature.delta(vector)
    #mfcc_delta_df = pd.DataFrame(mfcc_delta)

    #mfcc_delta2 = librosa.feature.delta(vector, order=2)
    #mfcc_delta2_df = pd.DataFrame(mfcc_delta2)

    
    #Features = pd.concat([vector_df, mfcc_delta_df,mfcc_delta2_df], axis=1)
    #vector_df = pd.DataFrame(Features)

    #vector_df.set_axis(list(range(0, 39)), axis='columns', inplace=True)
    #vector_df.to_csv("Vector_Test.csv")
    
    File_array = []
    Score_array = []
    Max_Array =["Max_Score"]

    path = "Models/"
    dirs = os.listdir(path)
    Testing_dir = source+user+"/"+"Testing/"

    for fileas in dirs:

        File_array.append(fileas) 
        trained_model = open(path+fileas,"rb")
        gmm_Model = pickle.load(trained_model)
        score = gmm_Model.score(Feature_DF)
        #prob = np.exp(score)

        Score_array = np.append(Score_array,score)
        #Prob_Array = np.append(Prob_Array,prob)
    
    File_array = pd.Series(File_array)    
    Score_array = pd.Series(Score_array)

    Max_value = max(Score_array)
    Max_Array.append(Max_value)
    #Min_array = pd.Series(Prob_Array)

    Df = pd.DataFrame([File_array,Score_array])
    #--------------------------------------------------


    i=i+1

    while i<ii:

        fs, sig = wav.read(source+user+"/"+"Testing/"+user+" ("+str(i)+").wav" )
        audio= silence_reduction.remove_silence(fs,sig)
        audio = logmmse.logmmse(audio,fs)
        #audio=sig
        #silence_reduction.remove_silence(sr,audio)
        #audio = logmmse.logmmse_from_file(source+user+"/"+"Training/"+user+" ("+str(i)+").wav")
        sig=audio
        file_name = user+" ("+str(i)+")"
        vector = mfcc(sig,fs)
        vector_df = pd.DataFrame(vector)
        #vector = librosa.feature.mfcc(sig,fs)
                

            
        mfcc_delta = librosa.feature.delta(vector)
        mfcc_delta_df = pd.DataFrame(mfcc_delta)

        mfcc_delta2 = librosa.feature.delta(vector, order=2)
        mfcc_delta2_df = pd.DataFrame(mfcc_delta2)

        
        Features = pd.concat([vector_df, mfcc_delta_df,mfcc_delta2_df], axis=1)
        FINAL_FEATURES = pd.DataFrame(Features)

        FINAL_FEATURES.set_axis(list(range(0, 39)), axis='columns', inplace=True)
        
        Feature_DF = pd.DataFrame(FINAL_FEATURES)


        
        Score_array = []

        for fileas in dirs:

            #File_array.append(fileas) 
            trained_model = open(path+fileas,"rb")
            gmm_Model = pickle.load(trained_model)
            score = gmm_Model.score(FINAL_FEATURES)
            #prob = np.exp(score)

            Score_array = np.append(Score_array,score)
            #Prob_Array = np.append(Prob_Array,prob)


        Score_array = pd.Series(Score_array)
        
        Max_value = max(Score_array)
        Max_Array.append(Max_value)
        

        Df=Df.append(Score_array,ignore_index=True)
        
        i=i+1

    mdf=pd.DataFrame(Max_Array)

    Df = pd.concat([Df, mdf], axis=1)
    Df.columns = Df.iloc[0]
    Df = Df[1:]

    #print(Df)
    Max = Df["Max_Score"]
    Current = Df[str(user)+".gmm"]

    #print(Max)
    #print("11111111111111111111")
    #print(Current)

    new_array = np.intersect1d(Max,Current)
    Correct=len(new_array)

    accuracy = (Correct/(ii-1))*100
    print(user,accuracy)
    Df.to_csv("Results/"+user+".csv")
