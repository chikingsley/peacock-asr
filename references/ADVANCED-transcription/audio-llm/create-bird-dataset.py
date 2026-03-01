import requests
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
import os

def fetch_bird_songs(bird_name, num_songs=10):
    base_url = "https://xeno-canto.org/api/2/recordings"
    query = f"gen:{bird_name}"  # Searching by genus
    bird_songs = []

    print(f"Fetching songs for {bird_name}...")
    for page in range(1, (num_songs // 10) + 2):  # Fetch multiple pages if needed
        response = requests.get(f"{base_url}?query={query}&page={page}")
        
        if response.status_code == 200:
            data = response.json()
            recordings = data.get("recordings", [])
            
            for recording in recordings:
                license_url = recording.get("lic")
                if "by" in license_url and "nc" not in license_url:  # Check for a suitable license
                    bird_songs.append({
                        "id": recording.get("id"),
                        "name": recording.get("en"),
                        "url": recording.get("file"),  # Keep the source URL
                        "license": license_url
                    })
                    if len(bird_songs) >= num_songs:  # Stop if we have enough songs
                        break
            time.sleep(1)  # Rate limit: wait for 1 second between requests
        else:
            print("Error fetching data:", response.json().get("error", {}).get("message"))
            return []

    print(f"Fetched {len(bird_songs)} songs for {bird_name}.")
    return bird_songs

# List of 20 bird names (you can modify this list as needed)
bird_names = [
    "Troglodytes", "Carduelis", "Parus", "Turdus", "Sturnus",
    "Alauda", "Emberiza", "Sylvia", "Phylloscopus", "Motacilla",
    "Cyanistes", "Fringilla", "Lanius", "Ficedula", "Hirundo",
    "Cisticola", "Acrocephalus", "Phoenicurus", "Anthus", "Buteo"
]

# Additional common birds to search for if needed
additional_bird_names = [
    "Corvus", "Passer", "Pica", "Sturnus", "Acanthis",
    "Hirundo", "Turdus", "Cyanocitta", "Picoides", "Dendrocopos"
]

all_bird_songs = []

# Fetch recordings for each bird
for bird in bird_names:
    songs = fetch_bird_songs(bird)
    if len(songs) >= 5:  # Only keep birds with at least 3 songs
        all_bird_songs.extend(songs)

# If not enough songs, try additional birds
if len(all_bird_songs) < 25:  # Adjust this threshold as needed
    print("Not enough songs found, searching additional common birds...")
    for bird in additional_bird_names:
        songs = fetch_bird_songs(bird)
        if len(songs) >= 5:  # Only keep birds with at least 3 songs
            all_bird_songs.extend(songs)

# Create a DataFrame
df = pd.DataFrame(all_bird_songs)

# Split the dataset into training and validation sets (90% train, 10% validation)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# Save the datasets to CSV files
train_df.to_csv("train_bird_songs.csv", index=False)
val_df.to_csv("val_bird_songs.csv", index=False)

print("Datasets prepared and saved as train_bird_songs.csv and val_bird_songs.csv.")

# Create a README file
readme_content = """
# Bird Songs Dataset

This dataset contains audio recordings of various bird species sourced from the Xeno-canto platform.

## Source of Data
The recordings were obtained from the Xeno-canto API, which provides access to a large collection of bird sounds from around the world.

## How It Was Created
The dataset was created by querying the Xeno-canto API for specific bird genera and filtering the results to include only those recordings that have a suitable license for commercial use. Each bird genus was queried to retrieve five recordings.

## License Type
The recordings included in this dataset are filtered to ensure they have licenses that allow for commercial use. Specifically, only recordings with licenses that do not include "nc" (non-commercial) were included.

## Dataset Structure
The dataset contains the following fields:
- `id`: The unique identifier for the recording.
- `name`: The English name of the bird species.
- `url`: The URL to download the audio file.
- `license`: The license type of the recording.
"""

with open("hf_README.md", "w") as readme_file:
    readme_file.write(readme_content)

# Push to Hugging Face Hub
dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(val_df)
})

# Push to Hugging Face Hub
dataset_dict.push_to_hub("Trelis/bird-songs", private=False)  # Set private=False if you want it public