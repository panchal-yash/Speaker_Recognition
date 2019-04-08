import pyaudio
import wave
import os
path = os.getcwd()  

def Recorder(audio_path):

    CHUNK = 1
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = audio_path

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print(int(RATE / CHUNK * RECORD_SECONDS))

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def record_user_audio():
    print("please speak a word into the microphone")
    user_name = input("Enter User_Name:")
    audio_path = 'Recordings/'+str(user_name)+'.wav'
    print("Hello My name is "+user_name+" and this is my Voice")
    Recorder(audio_path)
    print("done - result written to "+str(user_name)+".wav")
    return audio_path,user_name



def record_user_audio_testing():
    print("please speak a word into the microphone")
    user_name = raw_input("Enter User_Name For Testing: ")

    audio_path = 'trainingData/Tests/'+str(user_name)+'.wav'
    print("Hello My name is "+user_name+" and this is my Voice")
    Recorder(audio_path)
    print("done - result written to "+str(user_name)+".wav")
    return audio_path,user_name
