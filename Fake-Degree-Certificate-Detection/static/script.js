const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const dropText = document.getElementById('drop-text');
const analyzeBtn = document.getElementById('analyze-btn');
const loadingDiv = document.getElementById('loading');

let selectedFile = null;
let chartInstance = null;

// Initialize Bar Chart for Confidence Insights
function initChart() {
    const ctx = document.getElementById('confidenceChart').getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Client 1 (Kaggle)', 'Client 2 (Roboflow)', 'Global Model'],
            datasets: [{
                label: 'Prediction Confidence Score (%)',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(99, 102, 241, 0.4)', // Indigo
                    'rgba(139, 92, 246, 0.4)', // Purple
                    'rgba(16, 185, 129, 0.4)'  // Emerald
                ],
                borderColor: [
                    'rgba(99, 102, 241, 1)', 
                    'rgba(139, 92, 246, 1)', 
                    'rgba(16, 185, 129, 1)'
                ],
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: { borderDash: [5, 5] }
                },
                x: {
                    grid: { display: false }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return ` Confidence: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
}

// Drag & Drop Functionality
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        processFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
        processFile(fileInput.files[0]);
    }
});

function processFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }
    
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        dropText.style.display = 'none';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// API Integration
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Toggle States
    analyzeBtn.disabled = true;
    loadingDiv.style.display = 'block';

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Unknown server error');
        }
        
        updateDashboard(data);
    } catch (err) {
        console.error(err);
        alert(`Prediction failed: ${err.message}`);
    } finally {
        analyzeBtn.disabled = false;
        loadingDiv.style.display = 'none';
    }
});

function updateDashboard(data) {
    // Render the 3 card results
    renderCard('card-client1', data.client1);
    renderCard('card-client2', data.client2);
    renderCard('card-global', data.global);
    
    // Animate Chart.js Bar Chart
    const confidences = [
        data.client1.confidence, 
        data.client2.confidence, 
        data.global.confidence
    ];
    
    // Dynamic chart styling based on Reality vs Fake
    const bgColors = [data.client1, data.client2, data.global].map(d => 
        d.prediction === 'Real' ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)'
    );
    const borderColors = [data.client1, data.client2, data.global].map(d => 
        d.prediction === 'Real' ? 'rgba(16, 185, 129, 1)' : 'rgba(239, 68, 68, 1)'
    );
    
    chartInstance.data.datasets[0].data = confidences;
    chartInstance.data.datasets[0].backgroundColor = bgColors;
    chartInstance.data.datasets[0].borderColor = borderColors;
    chartInstance.update();
}

function renderCard(cardId, resultData) {
    const card = document.getElementById(cardId);
    const box = card.querySelector('.prediction-box');
    const lbl = card.querySelector('.pred-label');
    const conf = card.querySelector('.pred-conf');
    
    lbl.textContent = resultData.prediction;
    conf.textContent = `${resultData.confidence}% Confidence`;
    
    // Remove existing classes, apply specific class for styling overrides
    box.classList.remove('pred-real', 'pred-fake');
    if (resultData.prediction === 'Real') {
        box.classList.add('pred-real');
    } else {
        box.classList.add('pred-fake');
    }
}

// On Load
document.addEventListener('DOMContentLoaded', () => {
    initChart();
});
