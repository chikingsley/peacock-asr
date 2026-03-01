import requests
import time
import webbrowser

def fetch_bird_songs(bird_name, num_songs=4):
    base_url = "https://xeno-canto.org/api/2/recordings"
    query = f"gen:{bird_name}"  # Searching by genus
    bird_songs = []

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
                        "url": recording.get("file"),
                        "license": license_url
                    })
                    if len(bird_songs) >= num_songs:  # Stop if we have enough songs
                        break
            time.sleep(1)  # Rate limit: wait for 1 second between requests
        else:
            print("Error fetching data:", response.json().get("error", {}).get("message"))
            return []

    return bird_songs

# Example usage
bird_name = "Troglodytes"  # Example genus
songs = fetch_bird_songs(bird_name)

for song in songs:
    print(f"ID: {song['id']}, Name: {song['name']}, URL: {song['url']}, License: {song['license']}")
    # Open the audio file in the web browser for playback
    webbrowser.open(song['url'])