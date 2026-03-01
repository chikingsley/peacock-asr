import socket
import sys
import wave
import time
import struct
import io

def checkArg():
    if len(sys.argv) != 3:
        print("ERROR. Wrong number of arguments passed. System will exit. Next time please supply 2 arguments!")
        sys.exit()
    else:
        print("2 Arguments exist. We can proceed further")

def checkPort():
    if int(sys.argv[2]) <= 5000:
        print("Port number invalid. Port number should be greater than 5000. Next time enter valid port.")
        sys.exit()
    else:
        print("Port number accepted!")

checkArg()
host = sys.argv[1]
try:
    port = int(sys.argv[2])
except ValueError:
    print("Error. Exiting. Please enter a valid port number.")
    sys.exit()
checkPort()

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print("Client socket initialized")
except socket.error:
    print("Failed to create socket")
    sys.exit()

while True:
    message = input("Enter text to convert to speech (or 'exit' to quit): ")
    
    start_time = time.time()
    s.sendto(message.encode('utf-8'), (host, port))
    
    if message.lower() == "exit":
        print("Exiting client...")
        break
    
    print("Receiving audio data...")
    
    # Receive sample rate
    sample_rate_data, _ = s.recvfrom(4)
    sample_rate = struct.unpack('I', sample_rate_data)[0]
    
    audio_data = b""
    while True:
        chunk, _ = s.recvfrom(1024)
        if chunk == b"END":
            break
        audio_data += chunk
    
    receive_time = time.time()
    print(f"Time taken to receive audio data: {receive_time - start_time:.2f} seconds")
    
    print("Audio data received. Saving to file...")
    with wave.open("output.wav", "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    
    save_time = time.time()
    print(f"Time taken to save audio file: {save_time - receive_time:.2f} seconds")
    print("Audio saved to 'output.wav'")
    
    total_time = save_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Audio file size: {len(audio_data)} bytes")

print("Program will end now.")
s.close()