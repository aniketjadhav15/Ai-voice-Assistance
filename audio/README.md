# Voice-to-Text and Text-to-Speech Web Application

This project is a Flask-based web application that processes audio files to transcribe voice to text, generate a response using a language model (GPT-2), and convert the response back into speech. The processed audio is then available for playback on the client-side.

## Features

- **Voice-to-Text:** Converts spoken words in an uploaded audio file to text using Whisper.
- **Language Model Response:** Generates a meaningful response to the transcribed text using GPT-2.
- **Text-to-Speech:** Converts the generated text response back to speech and makes the audio available for download and playback.
- **Audio Speed Adjustment:** Optional functionality to adjust the speed of the final audio output.

## Project Structure

```
├── app.py                # Main Flask application
├── models
│   └── model_setup.py    # Model setup and utility functions for voice-to-text, LLM query, and text-to-speech
├── templates
│   └── index.html        # Frontend template for file upload and audio playback
├── audio                 # Directory for storing input and output audio files
│   ├── input_audio.wav   # Temporary storage for uploaded audio
│   └── output_audio.wav  # Generated audio response
├── static                # Static files (if any)
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/voice-to-text-tts-app.git
cd voice-to-text-tts-app
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Model Setup

Ensure you have all the necessary models downloaded:

- Whisper for voice-to-text.
- GPT-2 or another chosen language model for text generation.
- Tacotron2 or any other TTS model for text-to-speech.

### 5. Run the Application

```bash
python app.py
```

### 6. Access the Application

Open your web browser and navigate to `http://127.0.0.1:5000/` to access the application.

## Usage

1. **Upload an Audio File:** On the web page, click the "Choose File" button to upload an audio file (WAV format).
2. **Process the Audio:** The server will process the uploaded audio, transcribe the voice to text, generate a response using GPT-2, and convert the response to speech.
3. **Play the Generated Audio:** Once processing is complete, the generated audio response will be available for playback directly on the page.

## API Endpoints

- **GET /**: Renders the main page for uploading audio files.
- **POST /process_audio**: Processes the uploaded audio, transcribes it, generates a response, converts it to speech, and returns the results.
- **GET /audio/<filename>**: Serves audio files from the `audio` directory.

## Troubleshooting

- **Audio Playback Issues:** Ensure the audio file is correctly generated and the URL is properly set in the client-side JavaScript.
- **Model Errors:** Check that all required models are correctly installed and paths are properly set up in `model_setup.py`.

## Contributing

Feel free to submit issues or pull requests to help improve the project.

---

This `README.md` provides a comprehensive guide to setting up, running, and using the project. It includes installation instructions, an overview of the project structure, and usage details, ensuring that others can easily understand and contribute to the project.