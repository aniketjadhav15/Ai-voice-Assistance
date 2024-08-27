document.getElementById('upload-form').onsubmit = function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const fileField = document.getElementById('audio_file');
    
    formData.append('audio_file', fileField.files[0]);
    
    fetch('/process_audio', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        
        if (data.error) {
            document.getElementById('text-output').textContent = data.error;
            return;
        }

        document.getElementById('text-output').textContent = data.transcript + "\n" + data.response;

        // Set the audio source and display the audio player
        document.getElementById('audio-output').src = data.audio_url;
        document.getElementById('audio-output').style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('text-output').textContent = "Error processing the audio.";
    });
};
