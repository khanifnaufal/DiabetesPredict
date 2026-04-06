document.addEventListener('DOMContentLoaded', () => {
    const predictionForm = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.getElementById('loader');
    const resultContainer = document.getElementById('result-container');

    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI Feedback
        submitBtn.disabled = true;
        btnText.style.display = 'none';
        loader.style.display = 'block';
        resultContainer.style.display = 'none';

        // Collect form data
        const formData = new FormData(predictionForm);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            const result = await response.json();

            if (result.status === 'success') {
                displayResult(result);
            } else {
                alert('Analysis Error: ' + result.message);
            }
        } catch (error) {
            console.error('Fetch error:', error);
            alert('A connection error occurred. Please ensure the server is running.');
        } finally {
            submitBtn.disabled = false;
            btnText.style.display = 'block';
            loader.style.display = 'none';
        }
    });

    function displayResult(data) {
        const riskClass = data.prediction === 1 ? 'high-risk' : 'low-risk';
        const riskIcon = data.prediction === 1 ? 'fa-exclamation-triangle' : 'fa-check-circle';
        
        resultContainer.innerHTML = `
            <div class="result-card ${riskClass}">
                <div class="result-header">
                    <i class="fas ${riskIcon} fa-3x mb-3"></i>
                    <h2>${data.risk_status} Detected</h2>
                </div>
                <p>AI calculated probability: <strong>${data.probability}%</strong></p>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${data.probability}%"></div>
                </div>
                <p class="recommendation">${data.recommendation}</p>
            </div>
        `;
        
        resultContainer.style.display = 'block';
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
});
