#!/usr/bin/env python3
import os
from pydub import AudioSegment
import glob

def convert_m4a_to_wav():
    """
    Convert all m4a files in the current directory to wav format.
    Keeps the original m4a files intact.
    """
    # Get all m4a files in the data directory
    m4a_files = glob.glob("data/*.m4a")
    
    if not m4a_files:
        print("No m4a files found in the data directory.")
        return
    
    print(f"Found {len(m4a_files)} m4a files. Converting to wav...")
    
    for m4a_file in m4a_files:
        # Get the file name without the extension
        base_filename = os.path.basename(m4a_file)
        filename_without_ext = os.path.splitext(base_filename)[0]
        
        # Load the m4a file
        sound = AudioSegment.from_file(m4a_file, format="m4a")
        
        # Define the output wav file name
        wav_file = f"data/{filename_without_ext}.wav"
        
        # Export as wav
        sound.export(wav_file, format="wav")
        
        print(f"Converted {m4a_file} to {wav_file}")
    
    print("Conversion complete!")

if __name__ == "__main__":
    convert_m4a_to_wav() 