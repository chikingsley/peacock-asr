import os
from openai import OpenAI

# Check if the POD_ID environment variable is set
pod_id = os.getenv("POD_ID")

# If POD_ID is not defined, ask for it from the command line
if not pod_id:
    pod_id = input("Please enter your POD_ID: ")

client = OpenAI(api_key="EMPTY", base_url=f"https://{pod_id}-8000.proxy.runpod.net/v1/")

# Open the audio file
audio_file = open("../data/validation.mp3", "rb") # adjust this as needed

# Send the transcription request
transcript = client.audio.transcriptions.create(
    model="deepdml/faster-whisper-large-v3-turbo-ct2",  # Model to use; optionally swap this for a custom model
    file=audio_file,                                    # The audio file
    language="en",                                      # Language for transcription
    response_format="vtt"                               # Response format
)

# Print the transcription result
print(transcript)