import os
import re
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from TTS.api import TTS
from pydub import AudioSegment
import uvicorn
import logging
import io
from functools import lru_cache

# Initialize FastAPI app
app = FastAPI()

# Enable logging
logging.basicConfig(level=logging.INFO)

# Define available voices and their corresponding models
VALID_VOICES = {
    # Main American Female (Jenny)
    "main_american_female": "tts_models/en/jenny/jenny",  # Jenny model for the main character

    # Supporting female characters (using VCTK model with specific speaker IDs)
    "american_female_1": "tts_models/en/vctk/vits",  # American Female 1 (p305)
    "american_female_2": "tts_models/en/vctk/vits",  # American Female 2 (p335)
    "american_female_3": "tts_models/en/vctk/vits",  # American Female 3 (p339)
    "american_female_4": "tts_models/en/vctk/vits",  # American Female 4 (p343)
    "american_female_5": "tts_models/en/vctk/vits",  # American Female 5 (p347)
    "british_female": "tts_models/en/vctk/vits",    # British Female (p364)

    # Supporting male characters (using VCTK model with specific speaker IDs)
    "american_male_1": "tts_models/en/vctk/vits",  # American Male 1 (p307)
    "american_male_2": "tts_models/en/vctk/vits",  # American Male 2 (p312)
    "american_male_3": "tts_models/en/vctk/vits",  # American Male 3 (p313)
    "american_male_4": "tts_models/en/vctk/vits",  # American Male 4 (p318)
    "british_male": "tts_models/en/vctk/vits",     # British Male (p330)
}

# Define character-to-voice mapping with speaker IDs (for multi-speaker models)
CHARACTER_VOICE_MAPPING = {
    # Main American Female (Jenny)
    "main_american_female": {"voice": "main_american_female", "speaker_id": None},  # No speaker ID for Jenny

    # American Females (VCTK model with specific speaker IDs)
    "american_female_1": {"voice": "american_female_1", "speaker_id": "p305"},
    "american_female_2": {"voice": "american_female_2", "speaker_id": "p335"},
    "american_female_3": {"voice": "american_female_3", "speaker_id": "p339"},
    "american_female_4": {"voice": "american_female_4", "speaker_id": "p343"},
    "american_female_5": {"voice": "american_female_5", "speaker_id": "p347"},

    # British Female (VCTK model)
    "british_female": {"voice": "british_female", "speaker_id": "p364"},

    # American Males (VCTK model with specific speaker IDs)
    "american_male_1": {"voice": "american_male_1", "speaker_id": "p307"},
    "american_male_2": {"voice": "american_male_2", "speaker_id": "p312"},
    "american_male_3": {"voice": "american_male_3", "speaker_id": "p313"},
    "american_male_4": {"voice": "american_male_4", "speaker_id": "p318"},

    # British Male (VCTK model)
    "british_male": {"voice": "british_male", "speaker_id": "p330"},
}

# Function to split text into natural chunks
def split_text_naturally(text, max_length=300):
    """
    Splits text into chunks that sound natural by preserving context and avoiding abrupt cuts.
    """
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding the next sentence doesn't exceed the max length, add it
        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            # If the current chunk is not empty, add it to the chunks list
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Start a new chunk with the current sentence
            current_chunk = sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Function to load a TTS model (with caching)
@lru_cache(maxsize=2)  # Cache up to 2 models
def load_model(voice):
    """
    Loads the TTS model for the specified voice.
    """
    if voice not in VALID_VOICES:
        raise ValueError(f"Invalid voice: {voice}")

    model_name = VALID_VOICES[voice]
    return TTS(model_name=model_name, progress_bar=False, gpu=False)

# Define character-specific speeds
CHARACTER_SPEEDS = {
    "main_american_female": 1.5, 
    
}

# TTS generation endpoint (supports both POST and GET)
@app.api_route("/tts", methods=["POST", "GET"])
async def generate_tts(
    text: str = Query(default=None, description="Text to convert to speech"),
    character: str = Query(default="main_american_female", description="Character voice"),
    speed: float = Query(default=1.0, description="Speech speed"),
):
    # Validate character
    if character not in CHARACTER_VOICE_MAPPING:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid character: '{character}'. Valid options are: {', '.join(CHARACTER_VOICE_MAPPING.keys())}"
        )

    # Get voice and speaker ID for the character
    character_config = CHARACTER_VOICE_MAPPING[character]
    voice = character_config["voice"]
    speaker_id = character_config["speaker_id"]

    # Load the TTS model
    tts = load_model(voice)

    try:
        # Split the text into natural chunks
        text_chunks = split_text_naturally(text, max_length=300)
        audio_files = []

        # Generate audio for each chunk
        for i, chunk in enumerate(text_chunks):
            temp_file = f"temp_{i}.wav"
            if speaker_id:
                tts.tts_to_file(text=chunk, file_path=temp_file, speaker=speaker_id, speed=speed)
            else:
                tts.tts_to_file(text=chunk, file_path=temp_file, speed=speed)
            audio_files.append(temp_file)

        # Combine audio files
        combined_audio = AudioSegment.empty()
        for audio_file in audio_files:
            combined_audio += AudioSegment.from_wav(audio_file)

        # Stream the audio file
        audio_stream = io.BytesIO()
        combined_audio.export(audio_stream, format="wav")
        audio_stream.seek(0)  # Reset stream position

        return StreamingResponse(audio_stream, media_type="audio/wav")

    except Exception as e:
        logging.error(f"Error generating TTS: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating TTS: {str(e)}")

    finally:
        # Clean up temporary files
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                os.remove(audio_file)

# Run the app (for local testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)