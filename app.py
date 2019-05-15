from flask import Flask
import Training as Train
import Testing_Recorded as Test
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
import Record_audio_Fixed_Interval as rec


app = Flask(__name__)

#def upload_voice_samples(sample1,sample2,sample3):




def Enroll_Folder_creation(username):
    path_Train = "Dataset/%s/Training/"%username
    path_Test = "Dataset/%s/Testing/"%username     
    try:  
        os.makedirs(path_Train)  
        os.makedirs(path_Test)
    except OSError: 
        x = "Creation of the directory %s failed" % path_Train    
        y = "Creation of the directory %s failed" % path_Test
    else:  
        x = "Successfully created the directory %s & %s" %(path_Train,path_Test)
        #upload_voice_samples(sample1,sample2,sample3)
    return str(x) 


@app.route('/')
def index():  
    #x=enroll_training_data_upload()
    return "x" 

@app.route('/Training/<username>')
def Training(username):
    x = str(username)
    #Train.Train(x)
    Enroll_Folder_creation(username)
    #y = Training_data_upload(username)
    rec.Serverside_Recording_Training(username,1)
    rec.Serverside_Recording_Training(username,2)
    rec.Serverside_Recording_Training(username,3)
    Train.Train(x)
    return "Model Trained"
    #return "%s Model Trained Successfully.."% username

@app.route('/Testing/<username>')
def Testing(username):
    x = str(username)
    #speech_recording(username,1)
    rec.Serverside_Recording_Testing(username,1)
    a = Test.Testing(x)
    return str(a)


    

if __name__=='__main__':
    app.run(debug=True)

