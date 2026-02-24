document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const predictBtn = document.getElementById('predictBtn');
    const resultDiv = document.getElementById('result');
    const resultText = document.getElementById('resultText');
    const resultDetails = document.getElementById('resultDetails');

    // UI Feedback
    predictBtn.disabled = true;
    predictBtn.innerHTML = 'Analyzing...';
    resultDiv.classList.add('hidden');

    try {
        const formData = new FormData(this);
        const data = {};

        // Convert to proper types
        for (let [key, value] of formData.entries()) {
            // Check if it should be a number
            if (!isNaN(value) && value !== "") {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        }

        // Special manual parsing for selects or overrides if needed,
        // but parseFloat above should handle numeric inputs.
        // We need to ensure strings stay strings for categorical vars though.
        // The FormData loop will make numbers numbers if isNaN is false.
        // BUT categorical values like "0" or "1" strings vs number inputs.

        // Let's be explicit based on expected types from python script inspection
        const payload = {
            age: parseInt(document.getElementById('age').value),
            sex: document.getElementById('sex').value,
            chest_pain_type: document.getElementById('chest_pain_type').value,
            resting_blood_pressure: parseInt(document.getElementById('resting_blood_pressure').value),
            cholesterol: parseInt(document.getElementById('cholesterol').value),
            fasting_blood_sugar: document.getElementById('fasting_blood_sugar').value,
            rest_ecg: document.getElementById('rest_ecg').value,
            max_heart_rate: parseInt(document.getElementById('max_heart_rate').value),
            exercise_induced_angina: document.getElementById('exercise_induced_angina').value,
            oldpeak: parseFloat(document.getElementById('oldpeak').value),
            slope: document.getElementById('slope').value,
            vessels_colored_by_flourosopy: document.getElementById('vessels_colored_by_flourosopy').value,
            thalassemia: document.getElementById('thalassemia').value
        };

        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Server error: ${response.statusText}`);
        }

        const result = await response.json();

        if (result.error) {
            throw new Error(result.error);
        }

        // Display results
        resultDiv.classList.remove('hidden');

        if (result.class === 1) {
            resultText.innerText = "High Risk Detected";
            resultText.className = "danger";
            resultDetails.innerText = "The model suggests a high probability of heart disease based on the provided metrics. Please consult a cardiologist immediately.";
        } else {
            resultText.innerText = "Low Risk Detected";
            resultText.className = "success";
            resultDetails.innerText = "The model suggests a low probability of heart disease. Maintain a healthy lifestyle.";
        }

        // Scroll to result
        resultDiv.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        alert("An error occurred: " + error.message);
        console.error(error);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = '<span>Analyze Risk</span>';
    }
});

document.getElementById('resetBtn').addEventListener('click', function () {
    document.getElementById('predictionForm').reset();
    document.getElementById('result').classList.add('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
});
