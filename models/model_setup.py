import whisper
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from TTS.api import TTS
import io

# Load Whisper model for voice-to-text
whisper_model = whisper.load_model("base.en")

# Initialize the tokenizer and model for LLM (GPT-2)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# Set up the TTS model using the TTS library
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/en/ek1/tacotron2", progress_bar=False).to(device)

# Function to convert voice to text
def voice_to_text(audio_file_path):
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_file_path)

        # Resample audio to 16 kHz
        audio = audio.set_frame_rate(16000)

        # Save the resampled audio to a temporary file
        resampled_audio_path = "audio/temp_resampled.wav"
        audio.export(resampled_audio_path, format="wav")

        # Read the resampled audio file
        audio_data, sr = sf.read(resampled_audio_path)

        if sr != 16000:
            raise ValueError("Sampling rate must be 16 kHz")

        if audio_data.size == 0:
            raise ValueError("Audio data is empty")

        # Transcribe the audio using Whisper model
        result = whisper_model.transcribe(resampled_audio_path)
        return result['text']
    except Exception as e:
        print(f"Error in voice_to_text: {e}")
        return None

# Ensure that the pad_token is set for the tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token if it's not already set

# Function to query the LLM (GPT-2) for text generation
def query_llm(transcript_text):
    try:
        # Check if the input text is empty or only contains whitespace
        if not transcript_text.strip():
            return "Sorry, I could not generate a meaningful response."

        # Construct the prompt to simulate a conversational or assistant-like response
        prompt = f"User: {transcript_text}\nAssistant:"

        # Tokenize the input text
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Check if tokenization failed
        if inputs['input_ids'].size()[1] == 0:
            return "Sorry, I could not generate a meaningful response."

        # Generate response
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=150,  # Generate up to 150 new tokens
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,  # Enable sampling to introduce variety
            top_p=0.95,  # Use nucleus sampling
            temperature=0.7  # Control randomness
        )

        # Decode and clean the generated text
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Ensure the response has content and is relevant
        if response_text and response_text.lower() != transcript_text.lower():
            # Remove the prompt part from the response
            response_text = response_text.replace(prompt, '').strip()
            
            # Extract the first meaningful sentence or clause for TTS
            final_response = response_text.split('.')[0]
            return final_response
        else:
            return "Sorry, I could not generate a meaningful response."

    except Exception as e:
        print(f"Error in query_llm: {e}")
        return "Sorry, I could not generate a meaningful response."




# Function to convert text to speech and save as an audio file using the TTS library
def text_to_speech(text, output_file):
    try:
        # Split the text into smaller chunks if necessary (e.g., by sentence)
        # This is useful if the TTS model has a limit on the number of tokens or text length.
        chunks = text.split('. ')  # Split text by sentences for simplicity
        audio_segments = []

        for chunk in chunks:
            if chunk.strip():  # Ensure the chunk is not empty
                # Convert text chunk to speech
                tts_output = tts.tts(chunk.strip() + '.')
                
                # Convert to audio segment and append to the list
                with io.BytesIO() as audio_buffer:
                    sf.write(audio_buffer, tts_output, samplerate=22050, format='wav')
                    audio_buffer.seek(0)  # Rewind the buffer to the beginning
                    audio_segment = AudioSegment.from_file(audio_buffer, format='wav')
                    audio_segments.append(audio_segment)

        # Concatenate all audio segments into one
        final_audio = sum(audio_segments)

        # Export the final concatenated audio to the output file
        final_audio.export(output_file, format="mp3")
        print(f"Audio file saved to {output_file}")

    except Exception as e:
        print(f"Error in text_to_speech: {e}")

# Function to adjust audio speed
def adjust_audio(file_path, output_path, speed=1.0):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(file_path)
        original_duration = audio.duration_seconds
        
        # Check for very short audio (< 0.5 seconds) and pad if necessary
        if original_duration < 0.5:
            # Pad with silence before speed adjustment
            padded_audio = pad_audio(audio, target_duration=1.0)
            audio = padded_audio

        # Calculate the new frame rate for slowing down
        new_frame_rate = int(audio.frame_rate / speed)  # Lower the frame rate to slow down

        # Adjust the audio speed by resampling
        adjusted_audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})

        # Convert back to the original frame rate for proper playback
        adjusted_audio = adjusted_audio.set_frame_rate(audio.frame_rate)

        # Export the adjusted audio to an output file
        adjusted_audio.export(output_path, format="mp3")
        print(f"Adjusted audio file saved to {output_path}")

    except Exception as e:
        print(f"Error in adjust_audio: {e}")

def pad_audio(audio, target_duration=1.0):
    # Calculate padding duration
    duration = audio.duration_seconds
    padding_duration = target_duration - duration
    if padding_duration > 0:
        # Create silence audio
        silence = AudioSegment.silent(duration=padding_duration * 1000)  # Convert seconds to milliseconds
        # Append silence to original audio
        padded_audio = audio + silence
        return padded_audio
    return audio

def remove_padding(audio, original_duration):
    # This function should remove padding if it was added. For now, it just trims the audio to the original duration.
    if audio.duration_seconds > original_duration:
        end_trim_duration = (audio.duration_seconds - original_duration) * 1000  # Convert seconds to milliseconds
        trimmed_audio = audio[:-int(end_trim_duration)]
        return trimmed_audio
    return audio
