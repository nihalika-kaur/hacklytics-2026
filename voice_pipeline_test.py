# import os
# import time
# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
# from google.cloud import texttospeech
# from google.cloud import speech
# import google.genai as genai
# import simpleaudio as sa
# from pydub import AudioSegment
# import speech_recognition as sr
# from dotenv import load_dotenv

# # ==============================
# # LOAD API KEYS
# # ==============================
# load_dotenv()  # loads keys from .env file

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)

# # Set Google API key in environment
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# # ==============================
# # CONFIG
# # ==============================
# SAMPLE_RATE = 16000
# CHUNK_DURATION = 5  # seconds

# # ==============================
# # TTS
# # ==============================
# def speak(text):
#     print(f"[TTS] {text}")

#     client = texttospeech.TextToSpeechClient(client_options={"api_key": GOOGLE_API_KEY})

#     synthesis_input = texttospeech.SynthesisInput(text=text)
#     voice = texttospeech.VoiceSelectionParams(
#         language_code="en-US",
#         ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
#     )
#     audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

#     response = client.synthesize_speech(
#         input=synthesis_input, voice=voice, audio_config=audio_config
#     )

#     with open("response.mp3", "wb") as out:
#         out.write(response.audio_content)

#     audio = AudioSegment.from_mp3("response.mp3")
#     playback = sa.play_buffer(
#         audio.raw_data,
#         num_channels=audio.channels,
#         bytes_per_sample=audio.sample_width,
#         sample_rate=audio.frame_rate
#     )
#     playback.wait_done()

# # ==============================
# # RECORD AUDIO
# # ==============================
# def record_chunk(filename="input.wav"):
#     print(f"[Listening for {CHUNK_DURATION} seconds...]")
#     recording = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE),
#                        samplerate=SAMPLE_RATE,
#                        channels=1,
#                        dtype='int16')
#     sd.wait()
#     write(filename, SAMPLE_RATE, recording)
#     return filename

# # ==============================
# # SPEECH TO TEXT
# # ==============================
# def transcribe(file_path):
#     try:
#         # Use Google STT via API key
#         client = speech.SpeechClient(client_options={"api_key": GOOGLE_API_KEY})
#         with open(file_path, "rb") as f:
#             content = f.read()
#         audio = speech.RecognitionAudio(content=content)
#         config = speech.RecognitionConfig(
#             encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#             sample_rate_hertz=SAMPLE_RATE,
#             language_code="en-US"
#         )
#         response = client.recognize(config=config, audio=audio)
#         if not response.results:
#             return ""
#         transcript = response.results[0].alternatives[0].transcript
#         print(f"[User said] {transcript}")
#         return transcript
#     except Exception as e:
#         # Fallback: use SpeechRecognition locally
#         try:
#             r = sr.Recognizer()
#             with sr.AudioFile(file_path) as source:
#                 audio_data = r.record(source)
#             transcript = r.recognize_google(audio_data, language="en-US").strip()
#             print(f"[User said] {transcript} (fallback)")
#             return transcript
#         except Exception:
#             return ""

# # ==============================
# # GEMINI RESPONSE
# # ==============================
# def ask_gemini(user_text):
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     prompt = f"""
#     You are a driving safety assistant.
#     The driver said: "{user_text}"
#     Respond briefly (1-2 sentences).
#     Keep tone calm and safety-focused.
#     """
#     response = model.generate_content(prompt)
#     return response.text

# # ==============================
# # LISTEN LOOP
# # ==============================
# def listen_until_silence():
#     full_transcript = ""
#     while True:
#         file_path = record_chunk()
#         transcript = transcribe(file_path)
#         if transcript.strip() == "":
#             break
#         full_transcript += " " + transcript
#         print("[Still talking... extending 5 seconds]")
#     return full_transcript.strip()

# # ==============================
# # MAIN TEST FLOW
# # ==============================
# def fatigue_test_flow(fatigue_level):
#     if fatigue_level == "mild":
#         speak("You might be tired. Consider taking a short break.")
#     elif fatigue_level == "high":
#         speak("You seem extremely tired. Please consider pulling over safely.")

#     print("[Waiting for user response...]")
#     user_input = listen_until_silence()

#     if user_input == "":
#         print("[No response detected. Going dormant.]")
#         return

#     gemini_reply = ask_gemini(user_input)
#     speak(gemini_reply)

#     print("[Listening again after Gemini reply...]")
#     second_input = listen_until_silence()
#     if second_input:
#         second_reply = ask_gemini(second_input)
#         speak(second_reply)
#     else:
#         print("[Conversation ended. Dormant.]")

# # ==============================
# # RUN TEST
# # ==============================
# if __name__ == "__main__":
#     print("Testing HIGH fatigue scenario")
#     fatigue_test_flow("high")


# import os
# import time
# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
# from google.cloud import texttospeech
# from google.cloud import speech
# import google.generativeai as genai  # Fixed import
# import simpleaudio as sa
# from dotenv import load_dotenv

# # ==============================
# # LOAD API KEYS
# # ==============================
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Configure Gemini (Using the correct library now)
# genai.configure(api_key=GEMINI_API_KEY)

# # ==============================
# # CONFIG
# # ==============================
# SAMPLE_RATE = 16000
# CHUNK_DURATION = 5 

# # ==============================
# # TTS (Fixed: No FFmpeg/pydub needed)
# # ==============================
# def speak(text):
#     print(f"[SafeDrive] {text}")
#     client = texttospeech.TextToSpeechClient(client_options={"api_key": GOOGLE_API_KEY})

#     synthesis_input = texttospeech.SynthesisInput(text=text)
#     voice = texttospeech.VoiceSelectionParams(
#         language_code="en-US",
#         ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
#     )
#     # Using LINEAR16 (WAV) so we don't need ffmpeg to decode MP3
#     audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

#     response = client.synthesize_speech(
#         input=synthesis_input, voice=voice, audio_config=audio_config
#     )

#     # Play directly from memory
#     play_obj = sa.play_buffer(response.audio_content, 1, 2, 24000) # TTS usually returns 24kHz
#     play_obj.wait_done()

# # ==============================
# # RECORD AUDIO
# # ==============================
# def record_chunk(filename="input.wav"):
#     print(f"[Listening for {CHUNK_DURATION} seconds...]")
#     recording = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE),
#                        samplerate=SAMPLE_RATE,
#                        channels=1,
#                        dtype='int16')
#     sd.wait()
#     write(filename, SAMPLE_RATE, recording)
#     return filename

# # ==============================
# # SPEECH TO TEXT
# # ==============================
# def transcribe(file_path):
#     try:
#         client = speech.SpeechClient(client_options={"api_key": GOOGLE_API_KEY})
#         with open(file_path, "rb") as f:
#             content = f.read()
        
#         audio = speech.RecognitionAudio(content=content)
#         config = speech.RecognitionConfig(
#             encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#             sample_rate_hertz=SAMPLE_RATE,
#             language_code="en-US"
#         )
        
#         response = client.recognize(config=config, audio=audio)
        
#         if not response.results:
#             return ""
            
#         transcript = response.results[0].alternatives[0].transcript
#         print(f"[User] {transcript}")
#         return transcript
#     except Exception as e:
#         print(f"[Error in STT] {e}")
#         return ""

# # ==============================
# # GEMINI RESPONSE
# # ==============================
# def ask_gemini(user_text):
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     prompt = f"You are a driving safety assistant. The driver said: '{user_text}'. Respond briefly (1 sentence) to keep them engaged or safe."
#     response = model.generate_content(prompt)
#     return response.text

# # ==============================
# # LISTEN LOOP
# # ==============================
# def listen_until_silence():
#     full_transcript = ""
#     while True:
#         file_path = record_chunk()
#         transcript = transcribe(file_path)
#         if not transcript.strip():
#             break
#         full_transcript += " " + transcript
#         print("[Still talking... checking next 5s]")
#     return full_transcript.strip()

# # ==============================
# # MAIN TEST FLOW
# # ==============================
# def fatigue_test_flow(fatigue_level):
#     if fatigue_level == "mild":
#         speak("You might be tired. Consider taking a short break.")
#     elif fatigue_level == "high":
#         speak("You seem extremely tired. Please consider pulling over safely.")

#     print("[Waiting for user response...]")
#     user_input = listen_until_silence()

#     if not user_input:
#         print("[No response. Going dormant.]")
#         return

#     reply = ask_gemini(user_input)
#     speak(reply)

# if __name__ == "__main__":
#     print("--- SafeDrive Voice Pipeline Test ---")
#     fatigue_test_flow("high")


# import os
# import time
# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
# from google.cloud import texttospeech
# from google.cloud import speech
# import google.generativeai as genai  # FIX 1: Correct library import
# import simpleaudio as sa
# import speech_recognition as sr
# from dotenv import load_dotenv

# # ==============================
# # LOAD API KEYS
# # ==============================
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)

# # ==============================
# # CONFIG
# # ==============================
# SAMPLE_RATE = 16000
# CHUNK_DURATION = 5  # seconds

# # ==============================
# # TTS
# # ==============================
# def speak(text):
#     print(f"[TTS] {text}")

#     client = texttospeech.TextToSpeechClient(client_options={"api_key": GOOGLE_API_KEY})

#     synthesis_input = texttospeech.SynthesisInput(text=text)
#     voice = texttospeech.VoiceSelectionParams(
#         language_code="en-US",
#         ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
#     )
    
#     # FIX 2: Use LINEAR16 instead of MP3 to avoid pydub/ffmpeg entirely
#     audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

#     response = client.synthesize_speech(
#         input=synthesis_input, voice=voice, audio_config=audio_config
#     )

#     # Play back raw data using simpleaudio directly
#     play_obj = sa.play_buffer(response.audio_content, 1, 2, 24000)
#     play_obj.wait_done()

# # ==============================
# # RECORD AUDIO
# # ==============================
# def record_chunk(filename="input.wav"):
#     print(f"[Listening for {CHUNK_DURATION} seconds...]")
#     recording = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE),
#                        samplerate=SAMPLE_RATE,
#                        channels=1,
#                        dtype='int16')
#     sd.wait()
#     write(filename, SAMPLE_RATE, recording)
#     return filename

# # ==============================
# # SPEECH TO TEXT
# # ==============================
# def transcribe(file_path):
#     try:
#         # Use Google STT via API key
#         client = speech.SpeechClient(client_options={"api_key": GOOGLE_API_KEY})
#         with open(file_path, "rb") as f:
#             content = f.read()
#         audio = speech.RecognitionAudio(content=content)
#         config = speech.RecognitionConfig(
#             encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#             sample_rate_hertz=SAMPLE_RATE,
#             language_code="en-US"
#         )
#         response = client.recognize(config=config, audio=audio)
#         if not response.results:
#             return ""
#         transcript = response.results[0].alternatives[0].transcript
#         print(f"[User said] {transcript}")
#         return transcript
#     except Exception as e:
#         # Fallback: use SpeechRecognition locally
#         try:
#             r = sr.Recognizer()
#             with sr.AudioFile(file_path) as source:
#                 audio_data = r.record(source)
#             transcript = r.recognize_google(audio_data, language="en-US").strip()
#             print(f"[User said] {transcript} (fallback)")
#             return transcript
#         except Exception:
#             return ""

# # ==============================
# # GEMINI RESPONSE
# # ==============================
# def ask_gemini(user_text):
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     prompt = f"""
#     You are a driving safety assistant.
#     The driver said: "{user_text}"
#     Respond briefly (1-2 sentences).
#     Keep tone calm and safety-focused.
#     """
#     response = model.generate_content(prompt)
#     return response.text

# # ==============================
# # LISTEN LOOP
# # ==============================
# def listen_until_silence():
#     full_transcript = ""
#     while True:
#         file_path = record_chunk()
#         transcript = transcribe(file_path)
#         if transcript.strip() == "":
#             break
#         full_transcript += " " + transcript
#         print("[Still talking... extending 5 seconds]")
#     return full_transcript.strip()

# # ==============================
# # MAIN TEST FLOW
# # ==============================
# def fatigue_test_flow(fatigue_level):
#     if fatigue_level == "mild":
#         speak("You might be tired. Consider taking a short break.")
#     elif fatigue_level == "high":
#         speak("You seem extremely tired. Please consider pulling over safely.")

#     print("[Waiting for user response...]")
#     user_input = listen_until_silence()

#     if user_input == "":
#         print("[No response detected. Going dormant.]")
#         return

#     gemini_reply = ask_gemini(user_input)
#     speak(gemini_reply)

#     print("[Listening again after Gemini reply...]")
#     second_input = listen_until_silence()
#     if second_input:
#         second_reply = ask_gemini(second_input)
#         speak(second_reply)
#     else:
#         print("[Conversation ended. Dormant.]")

# # ==============================
# # RUN TEST
# # ==============================
# if __name__ == "__main__":
#     print("Testing HIGH fatigue scenario")
#     fatigue_test_flow("high")


# import os
# import time
# import sounddevice as sd
# import numpy as np
# from scipy.io.wavfile import write
# from google.cloud import texttospeech
# from google.cloud import speech
# import google.generativeai as genai # FIX 1: Correct library name for your code
# import simpleaudio as sa
# # FIX 2: Removed pydub so we don't need ffmpeg anymore
# import speech_recognition as sr
# from dotenv import load_dotenv

# # ==============================
# # LOAD API KEYS
# # ==============================
# load_dotenv()  # loads keys from .env file

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)

# # Set Google API key in environment
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# # ==============================
# # CONFIG
# # ==============================
# SAMPLE_RATE = 16000
# CHUNK_DURATION = 5  # seconds

# # ==============================
# # TTS
# # ==============================
# def speak(text):
#     print(f"[TTS] {text}")

#     client = texttospeech.TextToSpeechClient(client_options={"api_key": GOOGLE_API_KEY})

#     synthesis_input = texttospeech.SynthesisInput(text=text)
#     voice = texttospeech.VoiceSelectionParams(
#         language_code="en-US",
#         ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
#     )
    
#     # FIX 2: Request LINEAR16 (raw WAV) instead of MP3
#     audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)

#     response = client.synthesize_speech(
#         input=synthesis_input, voice=voice, audio_config=audio_config
#     )

#     # FIX 2: Play the raw bytes directly without needing pydub/ffmpeg
#     play_obj = sa.play_buffer(
#         response.audio_content,
#         num_channels=1,
#         bytes_per_sample=2,
#         sample_rate=24000 # Google TTS LINEAR16 default rate
#     )
#     play_obj.wait_done()

# # ==============================
# # RECORD AUDIO
# # ==============================
# def record_chunk(filename="input.wav"):
#     print(f"[Listening for {CHUNK_DURATION} seconds...]")
#     recording = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE),
#                        samplerate=SAMPLE_RATE,
#                        channels=1,
#                        dtype='int16')
#     sd.wait()
#     write(filename, SAMPLE_RATE, recording)
#     return filename

# # ==============================
# # SPEECH TO TEXT
# # ==============================
# def transcribe(file_path):
#     try:
#         # Use Google STT via API key
#         client = speech.SpeechClient(client_options={"api_key": GOOGLE_API_KEY})
#         with open(file_path, "rb") as f:
#             content = f.read()
#         audio = speech.RecognitionAudio(content=content)
#         config = speech.RecognitionConfig(
#             encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#             sample_rate_hertz=SAMPLE_RATE,
#             language_code="en-US"
#         )
#         response = client.recognize(config=config, audio=audio)
#         if not response.results:
#             return ""
#         transcript = response.results[0].alternatives[0].transcript
#         print(f"[User said] {transcript}")
#         return transcript
#     except Exception as e:
#         # Fallback: use SpeechRecognition locally
#         try:
#             r = sr.Recognizer()
#             with sr.AudioFile(file_path) as source:
#                 audio_data = r.record(source)
#             transcript = r.recognize_google(audio_data, language="en-US").strip()
#             print(f"[User said] {transcript} (fallback)")
#             return transcript
#         except Exception:
#             return ""

# # ==============================
# # GEMINI RESPONSE
# # ==============================
# def ask_gemini(user_text):
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     prompt = f"""
#     You are a driving safety assistant.
#     The driver said: "{user_text}"
#     Respond briefly (1-2 sentences).
#     Keep tone calm and safety-focused.
#     """
#     response = model.generate_content(prompt)
#     return response.text

# # ==============================
# # LISTEN LOOP
# # ==============================
# def listen_until_silence():
#     full_transcript = ""
#     while True:
#         file_path = record_chunk()
#         transcript = transcribe(file_path)
#         if transcript.strip() == "":
#             break
#         full_transcript += " " + transcript
#         print("[Still talking... extending 5 seconds]")
#     return full_transcript.strip()

# # ==============================
# # MAIN TEST FLOW
# # ==============================
# def fatigue_test_flow(fatigue_level):
#     if fatigue_level == "mild":
#         speak("You might be tired. Consider taking a short break.")
#     elif fatigue_level == "high":
#         speak("You seem extremely tired. Please consider pulling over safely.")

#     print("[Waiting for user response...]")
#     user_input = listen_until_silence()

#     if user_input == "":
#         print("[No response detected. Going dormant.]")
#         return

#     gemini_reply = ask_gemini(user_input)
#     speak(gemini_reply)

#     print("[Listening again after Gemini reply...]")
#     second_input = listen_until_silence()
#     if second_input:
#         second_reply = ask_gemini(second_input)
#         speak(second_reply)
#     else:
#         print("[Conversation ended. Dormant.]")

# # ==============================
# # RUN TEST
# # ==============================
# if __name__ == "__main__":
#     print("Testing HIGH fatigue scenario")
#     fatigue_test_flow("high")



import os
import io
import time
import sounddevice as sd
from scipy.io import wavfile
from scipy.io.wavfile import write
from google.cloud import texttospeech, speech
from google import genai
import speech_recognition as sr
from dotenv import load_dotenv

# ==============================
# LOAD API KEYS
# ==============================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ==============================
# INITIALIZE GEMINI CLIENT
# ==============================
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds
SILENCE_THRESHOLD = 2  # consecutive silent chunks before stopping listening

# ==============================
# TTS (Fixed for Windows Stability)
# ==============================
def speak(text):
    print(f"[TTS] {text}")
    client = texttospeech.TextToSpeechClient(client_options={"api_key": GOOGLE_API_KEY})
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

    # Use scipy and sounddevice to play the WAV data safely
    rate, audio_data = wavfile.read(io.BytesIO(response.audio_content))
    sd.play(audio_data, samplerate=rate)
    sd.wait()

# ==============================
# RECORD AUDIO
# ==============================
def record_chunk(filename="input.wav"):
    print(f"[Listening for {CHUNK_DURATION} seconds...]")
    recording = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype='int16')
    sd.wait()
    write(filename, SAMPLE_RATE, recording)
    return filename

# ==============================
# SPEECH TO TEXT
# ==============================
def transcribe(file_path):
    try:
        client = speech.SpeechClient(client_options={"api_key": GOOGLE_API_KEY})
        with open(file_path, "rb") as f:
            content = f.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="en-US"
        )
        response = client.recognize(config=config, audio=audio)
        if not response.results:
            return ""
        transcript = response.results[0].alternatives[0].transcript
        print(f"[User said] {transcript}")
        return transcript
    except Exception as e:
        # Fallback local recognition
        try:
            r = sr.Recognizer()
            with sr.AudioFile(file_path) as source:
                audio_data = r.record(source)
            transcript = r.recognize_google(audio_data, language="en-US").strip()
            print(f"[User said] {transcript} (fallback)")
            return transcript
        except:
            return ""

# ==============================
# GEMINI RESPONSE
# ==============================
def ask_gemini(user_text):
    response = gemini_client.models.generate_content(
        model="gemini-1.5-flash",
        contents=f"You are a driving safety assistant.\nDriver said: '{user_text}'.\nRespond in 1-2 sentences, calm and safety-focused."
    )
    return response.text

# ==============================
# LISTEN UNTIL USER STOPS TALKING
# ==============================
def listen_until_speech_and_stop():
    """
    Continuously record audio in 5s chunks.
    If speech is detected in a chunk, keep recording in additional 5s chunks.
    Stops after a silent chunk following speech.
    Returns full transcript.
    """
    full_transcript = ""
    has_spoken = False  # Flag: user has spoken at least once

    while True:
        file_path = record_chunk()  # always record at least 5s
        transcript = transcribe(file_path)

        if transcript.strip():
            full_transcript += " " + transcript
            has_spoken = True
            print("[Detected speech, adding another 5s chunk...]")
        else:
            if has_spoken:
                print("[No speech detected after last speaking chunk, stopping...]")
                break
            else:
                print("[No speech detected yet, waiting...]")
                # keep looping until user says something

    return full_transcript.strip()

# ==============================
# MAIN TEST FLOW
# ==============================
def fatigue_test_flow(fatigue_level):
    # Initial fatigue alert
    if fatigue_level == "mild":
        speak("You might be tired. Consider taking a short break.")
    elif fatigue_level == "high":
        speak("You seem extremely tired. Please consider pulling over safely.")

    # Fixed function call name
    print("[Waiting for user response...]")
    user_input = listen_until_speech_and_stop() 
    
    if not user_input:
        print("[No response detected. Going dormant.]")
        return

    # Send to Gemini
    gemini_reply = ask_gemini(user_input)
    speak(gemini_reply)

    # Listen again for follow-up (Fixed function call name)
    print("[Listening again after Gemini reply...]")
    follow_up = listen_until_speech_and_stop()
    
    if follow_up:
        gemini_followup = ask_gemini(follow_up)
        speak(gemini_followup)
    else:
        print("[Conversation ended. Dormant.]")

# ==============================
# RUN TEST
# ==============================
if __name__ == "__main__":
    print("--- SafeDrive Voice Pipeline Test ---")
    # Test HIGH fatigue scenario
    fatigue_test_flow("high")