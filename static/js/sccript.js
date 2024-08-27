document.getElementById('upload-form').onsubmit = function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const fileField = document.getElementById('audio_file');
    
    formData.append('audio_file', fileField.files[0]);
    
    fetch('/process_audio', {
        method: 'POST',
        body: formData
    }).then(response => response.blob())
    .then(blob => {
        const audioUrl = URL.createObjectURL(blob);
        document.getElementById('audio-output').src = audioUrl;
        document.getElementById('text-output').textContent = "Your audio has been processed.";
    }).catch(error => {
        console.error('Error:', error);
        document.getElementById('text-output').textContent = "Error processing the audio.";
    });
};
