document.addEventListener('DOMContentLoaded', function () {
    var form = document.getElementById('textToSketchForm');

    form.addEventListener('submit', function (event) {
        event.preventDefault();

        var textDescription = document.getElementById('textDescription').value;
        var sketchContainer = document.getElementById('sketchImage');
        sketchContainer.innerHTML = '<div class="loader"></div>';

        fetch('/generate_sketch', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text_description: textDescription })
        })
        .then(response => response.json())
        .then(data => {
            if(data.error) {
                sketchContainer.innerHTML = '<p class="error-message">Error: ' + data.error + '</p>';
            } else {
                var sketchBase64 = data.sketch;
                sketchContainer.innerHTML = '<img src="data:image/png;base64,' + sketchBase64 + '" alt="Generated Sketch" class="sketch-image">';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            sketchContainer.innerHTML = '<p class="error-message">An error occurred while generating the sketch.</p>';
        });
    });
});
