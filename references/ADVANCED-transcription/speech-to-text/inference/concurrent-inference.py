import os
import time
import asyncio
from openai import OpenAI

# Check if the POD_ID environment variable is set
pod_id = os.getenv("POD_ID")

# If POD_ID is not defined, ask for it from the command line
if not pod_id:
    pod_id = input("Please enter your POD_ID: ")

client = OpenAI(api_key="EMPTY", base_url=f"https://{pod_id}-8000.proxy.runpod.net/v1/")

# Path to the audio file (adjust this as needed)
audio_file_path = "../data/validation.mp3"

async def send_request(request_id):
    # Open the audio file in binary mode
    with open(audio_file_path, "rb") as audio_file:
        # Measure the response time
        start_time = time.time()

        # Send the request using the OpenAI library
        transcript = client.audio.transcriptions.create(
            model="deepdml/faster-whisper-large-v3-turbo-ct2",
            file=audio_file,
            language="en",
            response_format="vtt"
        )

        end_time = time.time()
        response_time = end_time - start_time

        print(f"Request {request_id}: Response time: {response_time:.4f} seconds")
        print(f"Response: {transcript}")

async def main():
    tasks = []
    for i in range(2):  # 2 concurrent requests
        task = asyncio.ensure_future(send_request(i + 1))
        tasks.append(task)
        await asyncio.sleep(0.01)  # 10 ms delay between requests

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
