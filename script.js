document.addEventListener('DOMContentLoaded', () => {
  let currentPredictionData = {};
  let currentPredictionResult = null;

  const predictBtn = document.querySelector('.predict-btn');
  const resultValue = document.getElementById('result-value');
  const feedbackSection = document.getElementById('feedback-section');
  const submitFeedbackBtn = document.getElementById('submit-feedback-btn');
  const actualTimeInput = document.getElementById('actual-time');
  const thanksMessage = document.getElementById('feedback-thanks');

  // --- Slider Value Display ---
  const sliders = [
    { id: 'distance', valueId: 'distance-value', unit: 'km' },
    { id: 'preparation', valueId: 'preparation-value', unit: 'min' },
    { id: 'experience', valueId: 'experience-value', unit: 'yrs' },
  ];

  sliders.forEach(({ id, valueId, unit }) => {
    const slider = document.getElementById(id);
    const valueDisplay = document.getElementById(valueId);
    valueDisplay.textContent = `${slider.value} ${unit}`;
    slider.addEventListener('input', () => {
      valueDisplay.textContent = `${slider.value} ${unit}`;
    });
  });

  // --- Button Group (Single-select) ---
  document.querySelectorAll('.button-group').forEach(group => {
    group.addEventListener('click', (e) => {
      if (e.target.classList.contains('btn-option')) {
        group.querySelectorAll('.btn-option').forEach(btn => btn.classList.remove('active'));
        e.target.classList.add('active');
      }
    });
  });

  // --- Weather Button Toggle (Multi-select) ---
  document.querySelector('.weather-group').addEventListener('click', (e) => {
    if (e.target.classList.contains('btn-weather')) {
      e.target.classList.toggle('active');
    }
  });

  // --- Prediction Logic ---
  predictBtn.addEventListener('click', async (e) => {
    e.preventDefault();

    resultValue.textContent = '...';
    predictBtn.disabled = true;
    feedbackSection.classList.add('hidden');
    thanksMessage.classList.add('hidden');

    const data = {
      Distance: parseFloat(document.getElementById('distance').value),
      Preparation_Time: parseFloat(document.getElementById('preparation').value),
      Courier_Experience: parseFloat(document.getElementById('experience').value),
      Weather_Foggy: isActiveWeather('foggy'),
      Weather_Rainy: isActiveWeather('rainy'),
      Weather_Snowy: isActiveWeather('snowy'),
      Weather_Windy: isActiveWeather('windy'),
      Traffic_Level_Low: isActiveOption('traffic', 'low'),
      Traffic_Level_Medium: isActiveOption('traffic', 'medium'),
      Time_of_Day_Evening: isActiveOption('time', 'evening'),
      Time_of_Day_Morning: isActiveOption('time', 'morning'),
      Time_of_Day_Night: isActiveOption('time', 'night'),
      Vehicle_Type_Car: isActiveOption('vehicle', 'car'),
      Vehicle_Type_Scooter: isActiveOption('vehicle', 'scooter'),
    };

    currentPredictionData = data;

    try {
      // Use the variable from config.js
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!response.ok) throw new Error('Prediction request failed');

      const result = await response.json();
      currentPredictionResult = result.predicted_delivery_time;
      resultValue.textContent = `${currentPredictionResult} min`;
      feedbackSection.classList.remove('hidden');

    } catch (error) {
      console.error('Prediction failed:', error);
      resultValue.textContent = 'Error';
    } finally {
      predictBtn.disabled = false;
    }
  });

  // --- Feedback Submission + Retraining Logic ---
  submitFeedbackBtn.addEventListener('click', async () => {
    const actualTime = actualTimeInput.value;
    if (!actualTime || isNaN(actualTime)) {
      alert('Please enter the actual delivery time.');
      return;
    }

    const feedbackPayload = {
      ...currentPredictionData,
      predicted_delivery_time: currentPredictionResult,
      actual_delivery_time: parseFloat(actualTime),
    };

    try {
      // Use the variable from config.js
      const response = await fetch(`${API_BASE_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedbackPayload),
      });

      if (!response.ok) throw new Error('Feedback submission failed');

      thanksMessage.classList.remove('hidden');
      feedbackSection.classList.add('hidden');
      actualTimeInput.value = '';

      // 🔁 Check if retrain should trigger
      try {
        // Use the variable from config.js
        const statsRes = await fetch(`${API_BASE_URL}/feedback-stats`);
        const statsData = await statsRes.json();

        if (statsData.total_feedback_count > 9) {
          console.log('📊 Triggering retraining...');
          // Use the variable from config.js
          const retrainRes = await fetch(`${API_BASE_URL}/retrain`, {
            method: 'POST',
          });
          const retrainResult = await retrainRes.json();
          console.log('✅ Retraining result:', retrainResult);
        }
      } catch (error) {
        console.warn('⚠️ Failed to check retrain trigger:', error);
      }

    } catch (error) {
      console.error('Feedback submission failed:', error);
      alert('Could not submit feedback. Please try again.');
    }
  });

  // --- Helpers ---
  function isActiveOption(group, value) {
    const btn = document.querySelector(`.btn-option[data-group="${group}"][data-value="${value}"]`);
    return btn && btn.classList.contains('active') ? 1 : 0;
  }

  function isActiveWeather(type) {
    const btn = document.querySelector(`.btn-weather[data-weather="${type}"]`);
    return btn && btn.classList.contains('active') ? 1 : 0;
  }
});