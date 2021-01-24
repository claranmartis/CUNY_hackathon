#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the REST API for async
batch processing.

Example usage:
    python transcribe_async.py resources/audio.raw
    python transcribe_async.py gs://cloud-samples-tests/speech/vr.flac
"""

import io
import os
from pygame import time

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "D:\CUNYhackathon\CUNYhackathon-b95ae2fbc665.json"


video_lib = ['Slide_1', 'Slide_2', 'Slide_3', 'Slide_4', 'Slide_5', 'Slide_6']

text_lib = ['One Tire', 'Makes A Swing', 'Two Pieces Of Bread', 'Make A Sandwich', 'Three Snowballs', 'Make A Snowman']


# [START speech_transcribe_async]
def transcribe_file(speech_file):
    """Transcribe the given audio file asynchronously."""
    from google.cloud import speech

    client = speech.SpeechClient()

    # [START speech_python_migration_async_request]
    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    """
     Note that transcription is limited to a 60 seconds audio file.
     Use a GCS file for audio longer than 1 minute.
    """
    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US",
    )

    # [START speech_python_migration_async_response]

    operation = client.long_running_recognize(config=config, audio=audio)
    # [END speech_python_migration_async_request]

    print("Waiting for operation to complete...")
    response = operation.result(timeout=90)
    
    converted_text = []

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        transcribed = result.alternatives[0].transcript
        converted_text.append(transcribed)
        print(u"Transcript: {}".format(transcribed))
        print("Confidence: {}".format(result.alternatives[0].confidence))
    # [END speech_python_migration_async_response]
    
    return_string = ' '.join(converted_text)
    return(return_string)


# [END speech_transcribe_async]




def synthesize_text(text, outfile):
    """Synthesizes speech from the input string of text."""
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Standard-C",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    #make sure the file is not open
    try:
        os.remove(outfile)
        print('overwriting audio file')
    except:
        pass
    
    
    # The response's audio_content is binary.
    with open(outfile, "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file ' + outfile)

#count occurances of each word
def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts


#input two strings to compare
def compare_strings(ref, in_text):

    #remove pounctuations from text
    import string
    try:
        in_text.translate(str.maketrans('','', string.punctuation))
    except:
        pass

    #convert both strings to uppercase for comparision
    ref.upper()
    in_text.upper()
    
    print(ref)
    print(in_text)

    
    ref_words = word_count(ref)
    in_text_words = word_count(in_text)
    
    return(ref_words.keys() - in_text_words.keys())


#plays .mp3 files
def playback(filename):
    
    from playsound import playsound
    playsound(filename, True)
    '''
    from pygame import mixer
    
    mixer.init()
    mixer.music.load(filename)
    mixer.music.play()
    #while mixer.music.get_busy(): # check if the file is playing
        #pass
        #time.wait(2000)
        #print('player busy')
    mixer.quit()
    #os.remove(filename)'''
    return('played audio')
    

#Record audio as wav file
def record_wav():
        
    import sounddevice as sd
    from scipy.io.wavfile import write
    
    fs = 44100  # Sample rate
    seconds = 10  # Duration of recording
    filename = 'recorded.wav'
    
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, myrecording)  # Save as WAV file 
    
    return filename


#Video player
def play_video(filename):
    import cv2
    import numpy as np
    #ffpyplayer for playing audio
    from ffpyplayer.player import MediaPlayer
    video_path = filename
    def PlayVideo(video_path):
        video=cv2.VideoCapture(video_path)
        player = MediaPlayer(video_path)
        while True:
            grabbed, frame=video.read()
            audio_frame, val = player.get_frame()
            if not grabbed:
                print("End of video")
                break
            if cv2.waitKey(28) & 0xFF == ord("q"):
                break
            cv2.imshow("Video", frame)
            if val != 'eof' and audio_frame is not None:
                #audio
                img, t = audio_frame
        video.release()
        cv2.destroyAllWindows()
    PlayVideo(video_path)

if __name__ == "__main__":
    
    '''play_video('Pizza_Song.mp4')
    #infile = '.\commercial_mono.wav'
    infile = 'Twinkle.wav'
    #infile = record_wav()
    in_text = transcribe_file(infile)
    
    ref = 'Twinkle Twinkle Little Star'
    '''    
    infile = 'Twinkle.wav'
    play_video('Intro.mp4')
    
    for v,t in zip(video_lib, text_lib):
        play_video('videos/'+v+'.mp4')
        ref = t
        print(ref)
        
        in_text = transcribe_file(infile)
        diff = compare_strings(ref, in_text)
        print(diff)
        
        if(diff != None):
            for word in diff:
                outfile = word+".mp3"
                print('output filename: '+ outfile)
                speak = 'say ' + word
                synthesize_text(speak, outfile)
                time.wait(2000)
                print(playback(outfile))
                
        else:
            print('Good Job!')
        
    
    
    

