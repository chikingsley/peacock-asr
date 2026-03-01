# Melo TTS Inferencing Setup

Allows you to:
- Inference Melo TTS on a remote server, via TCP or UDP.
- Fine-tuning is supported in the MeloTTS repo, but not covered here.

## SERVER SETUP

Start with:
```
pip install uv
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
uv venv
source .venv/bin/activate
```

Adjust the `requirements.txt` file to remove the `transformers` version.

```
apt-get update -y
apt-get install build-essential gcc -y
apt-get install portaudio19-dev -y
uv pip install -r requirements.txt
uv pip install fastapi uvicorn requests unidic pyaudio
uv run --with unidic -- python -m unidic download
```

Simple inference test:
```
melo "hi" output.wav
```
Note: the above command didn't work. Unclear why.

> Open Python CLI and install required nltk dependency for punctation marks:
```
import nltk
nltk.download('averaged_perceptron_tagger_eng')
```

Now put `Server.py` and `utils.py` in the cloned MeloTTS directory and run:
```
uv run -- python Server.py
```


## LOCAL SETUP
Run the `client.py` file to make requests to the server.

```python
pip install uv
uv venv
source .venv/bin/activate
uv pip install requests numpy scipy
uv run -- python client.py
```
(0.7s latency on an RTX A6000 in the UK, from Dublin)


# TO GET GOAT LEVEL SPEED Follow the Steps Below:

## UDP Server Setup

1. Place `Serverudp.py` and `utils.py` in the MeloTTS directory.
2. Install additional dependencies:
   ```bash
uv pip install numpy scipy pyaudio
   ```
3. Run the UDP server:
   ```bash
uv run -- python Serverudp.py 8082
   ```
   Replace `<port_number>` with a udp port number greater than 5000.


## For UDP client:
1. Place `Clientudp.py` in your local directory.
2. Run the UDP client:
   ```bash
uv run -- python Clientudp.py 149.36.1.168 47120
   ```
   Replace `<server_ip>` with the IP address of your server and `<port_number>` with the external mapped port number used for the server.
   
For example, in the image below, the external port mapping for 8082 udp port is 39196 (find the wide blue button on the top of a Vast AI instance):

<img width="200" alt="Screenshot 2024-08-24 at 12 54 01 PM" src="https://github.com/user-attachments/assets/3f21457e-b6d5-4fc9-89d9-e795e8cec799">

So, you would use, 
```python
python Clientudp.py 10.167.17.8 39196
```

## UDP Usage Notes

- The UDP server uses MeloTTS to generate speech from text.
- The UDP client sends text to the server and receives audio data in return.
- Audio is streamed in chunks, allowing for real-time playback.
- To exit the client or server, type "exit" as the input text.

Here is some sample output:
```
uv run -- python Clientudp.py 149.36.1.168 47120
2 Arguments exist. We can proceed further
Port number accepted!
Client socket initialized
Enter text to convert to speech (or 'exit' to quit): How's life Rohan?
Receiving audio data...
Time taken to receive audio data: 3.54 seconds
Audio data received. Saving to file...
Time taken to save audio file: 0.00 seconds
Audio saved to 'output.wav'
Total time taken: 3.54 seconds
Audio file size: 166280 bytes
Enter text to convert to speech (or 'exit' to quit): How's life Rohan?
Receiving audio data...
Time taken to receive audio data: 0.20 seconds
Audio data received. Saving to file...
Time taken to save audio file: 0.00 seconds
Audio saved to 'output.wav'
Total time taken: 0.20 seconds
Audio file size: 169352 bytes
Enter text to convert to speech (or 'exit' to quit): Life is really great!
Receiving audio data...
Time taken to receive audio data: 0.22 seconds
Audio data received. Saving to file...
Time taken to save audio file: 0.00 seconds
Audio saved to 'output.wav'
Total time taken: 0.22 seconds
Audio file size: 206216 bytes
```



