async function predict() {
    const features = document.getElementById('features').value.split(',').map(Number);
    
    // Send features to backend for prediction
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ features })
    });

    const result = await response.json();
    displayResult(result.prediction);
}

function displayResult(prediction) {
    const resultContainer = document.getElementById('result-container');
    resultContainer.innerHTML = prediction === 0 ? 'The tumor is Malignant' : 'The tumor is Benign';
}
