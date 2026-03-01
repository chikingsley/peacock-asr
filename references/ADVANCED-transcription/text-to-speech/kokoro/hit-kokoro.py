from openai import OpenAI

client = OpenAI(
    base_url="http://213.181.105.225:10821/v1",
    api_key="not-needed",
)

chosen_voice = "am_michael" # try also US female af_heart | af_bella, US male am_michael, UK female bf_emma, UK male bm_george

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice=chosen_voice,
    input=f"Greetings from Trelis, I'm {chosen_voice.split('_')[-1]}!",
) as resp:
    resp.stream_to_file(f"{chosen_voice}.mp3")