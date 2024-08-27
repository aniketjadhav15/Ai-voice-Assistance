from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from models.model_setup import voice_to_text, query_llm, text_to_speech, tts

app = Flask(__name__)

# Ensure the folder exists for storing audio files
os.makedirs('audio', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # Save the uploaded audio
        audio_file = request.files['audio_file']
        input_audio_path = 'audio/input_audio.wav'
        os.makedirs(os.path.dirname(input_audio_path), exist_ok=True)
        audio_file.save(input_audio_path)

        # Convert voice to text
        transcript = voice_to_text(input_audio_path)
        print(f"Transcript Text: {transcript}")
        if transcript is None:
            return jsonify({"error": "Error processing audio"}), 500

        # Query LLM for a response
        response_text = query_llm(transcript)
        if not response_text:
            return jsonify({"error": "No response from LLM"}), 500

        print("Response Text:", response_text)  # Debug: Print the entire response text

        # Convert text to speech
        output_audio_path = 'audio/output_audio.wav'
        
        # Use TTS to convert the entire response text to speech
        text_to_speech(response_text, output_audio_path)

        # Verify that the audio file has been saved
        if not os.path.exists(output_audio_path):
            return jsonify({"error": "Failed to save the audio file"}), 500

        # Return the processed audio file URL and response text
        return jsonify({
            "audio_url": f'/audio/output_audio.wav',  # Ensure correct URL path for the client
            "response": response_text,
            "transcript": transcript
        })

    except Exception as e:
        print(f"Error in process_audio: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('audio', filename)


if __name__ == '__main__':
    app.run(debug=True)
