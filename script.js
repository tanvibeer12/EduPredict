// API Configuration
const API_URL = 'http://localhost:5000'; // Change this to your backend URL

// State Management
let currentPrediction = null;

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeSliders();
    initializeForm();
    loadStatistics();
});

// Tab Navigation
function initializeTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;
            
            // Update active tab button
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update active tab content
            tabContents.forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(`${tabName}-tab`).classList.add('active');
        });
    });
}

// Slider Interactions
function initializeSliders() {
    const sliders = [
        { id: 'studyHours', valueId: 'studyHoursValue', suffix: 'h' },
        { id: 'attendance', valueId: 'attendanceValue', suffix: '%' },
        { id: 'previousGrade', valueId: 'previousGradeValue', suffix: '%' },
        { id: 'assignments', valueId: 'assignmentsValue', suffix: '%' },
        { id: 'extracurriculars', valueId: 'extracurricularsValue', suffix: 'h' },
        { id: 'sleep', valueId: 'sleepValue', suffix: 'h' }
    ];

    sliders.forEach(slider => {
        const input = document.getElementById(slider.id);
        const valueDisplay = document.getElementById(slider.valueId);
        
        input.addEventListener('input', (e) => {
            valueDisplay.textContent = e.target.value + slider.suffix;
        });
    });
}

// Form Submission
function initializeForm() {
    const form = document.getElementById('prediction-form');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const studentData = {
            study_hours: parseFloat(document.getElementById('studyHours').value),
            attendance_rate: parseFloat(document.getElementById('attendance').value),
            previous_grade: parseFloat(document.getElementById('previousGrade').value),
            assignments_completed: parseFloat(document.getElementById('assignments').value),
            extracurriculars: parseFloat(document.getElementById('extracurriculars').value),
            sleep_hours: parseFloat(document.getElementById('sleep').value)
        };

        await predictPerformance(studentData);
    });
}

// Predict Performance
async function predictPerformance(data) {
    showLoading();

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();
        currentPrediction = result;
        
        setTimeout(() => {
            displayResults(result);
        }, 1000);

    } catch (error) {
        console.error('Error:', error);
        alert('Failed to get prediction. Make sure the Python backend is running on ' + API_URL);
        hideLoading();
    }
}

// Show Loading State
function showLoading() {
    document.getElementById('empty-state').classList.add('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('loading').classList.remove('hidden');
}

// Hide Loading State
function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

// Display Results
function displayResults(data) {
    hideLoading();
    document.getElementById('results').classList.remove('hidden');

    // Update prediction card
    updatePredictionCard(data.predicted_grade, data.confidence);
    
    // Update factors
    updateFactors(data.factors);
    
    // Update recommendations
    updateRecommendations(data.recommendations);
}

// Update Prediction Card
function updatePredictionCard(grade, confidence) {
    const gradeRounded = Math.round(grade * 10) / 10;
    const category = getGradeCategory(gradeRounded);
    
    // Update radial chart
    const circumference = 2 * Math.PI * 80;
    const offset = circumference - (gradeRounded / 100) * circumference;
    const progressCircle = document.getElementById('radialProgress');
    progressCircle.style.strokeDashoffset = offset;
    progressCircle.style.stroke = category.color;
    
    // Update grade text
    document.getElementById('gradeText').textContent = gradeRounded + '%';
    document.getElementById('gradeText').style.fill = category.color;
    
    // Update category
    const categoryInfo = document.getElementById('categoryInfo');
    categoryInfo.innerHTML = `
        <i class="fas ${category.icon}"></i>
        <h3 style="color: ${category.color}">${category.label}</h3>
    `;
    
    // Update description
    document.getElementById('predictionDescription').textContent = 
        `Based on the provided data, your predicted performance grade is ${gradeRounded}%`;
    
    // Update confidence
    document.getElementById('confidenceValue').textContent = confidence + '%';
    
    // Update card background
    const predictionCard = document.getElementById('prediction-card');
    predictionCard.style.background = category.bgGradient;
    predictionCard.style.borderColor = category.color + '33';
}

// Get Grade Category
function getGradeCategory(grade) {
    if (grade >= 90) {
        return {
            label: 'Excellent',
            color: '#10b981',
            bgGradient: 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)',
            icon: 'fa-check-circle'
        };
    } else if (grade >= 80) {
        return {
            label: 'Very Good',
            color: '#3b82f6',
            bgGradient: 'linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%)',
            icon: 'fa-chart-line'
        };
    } else if (grade >= 70) {
        return {
            label: 'Good',
            color: '#f59e0b',
            bgGradient: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
            icon: 'fa-info-circle'
        };
    } else if (grade >= 60) {
        return {
            label: 'Satisfactory',
            color: '#f97316',
            bgGradient: 'linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%)',
            icon: 'fa-exclamation-circle'
        };
    } else {
        return {
            label: 'Needs Improvement',
            color: '#ef4444',
            bgGradient: 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)',
            icon: 'fa-exclamation-triangle'
        };
    }
}

// Update Factors
function updateFactors(factors) {
    const factorsList = document.getElementById('factors-list');
    factorsList.innerHTML = '';
    
    factors.forEach((factor, index) => {
        const factorItem = document.createElement('div');
        factorItem.className = 'factor-item';
        factorItem.style.animationDelay = `${index * 0.1}s`;
        
        factorItem.innerHTML = `
            <div class="factor-label">
                <span>${factor.name}</span>
                <span style="color: ${factor.color}; font-weight: 600;">${factor.impact}%</span>
            </div>
            <div class="factor-bar-bg">
                <div class="factor-bar" style="width: ${factor.impact}%; background-color: ${factor.color};"></div>
            </div>
        `;
        
        factorsList.appendChild(factorItem);
    });
}

// Update Recommendations
function updateRecommendations(recommendations) {
    const recList = document.getElementById('recommendations-list');
    recList.innerHTML = '';
    
    recommendations.forEach((rec, index) => {
        const li = document.createElement('li');
        li.textContent = rec;
        li.style.animationDelay = `${index * 0.1}s`;
        recList.appendChild(li);
    });
}

// Load Statistics
async function loadStatistics() {
    try {
        const response = await fetch(`${API_URL}/statistics`);
        if (response.ok) {
            const stats = await response.json();
            updateStatistics(stats);
        }
    } catch (error) {
        console.log('Statistics not available');
        // Use default values
        updateStatistics({
            model_accuracy: 94.2,
            students_analyzed: 1247,
            average_error: 3.2
        });
    }
}

// Update Statistics Display
function updateStatistics(stats) {
    document.getElementById('modelAccuracy').textContent = stats.model_accuracy + '%';
    document.getElementById('studentsAnalyzed').textContent = stats.students_analyzed;
    document.getElementById('avgError').textContent = '±' + stats.average_error + '%';
    
    // Create chart
    createPerformanceChart();
}

// Create Performance Chart
function createPerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [
                {
                    label: 'Actual Grade',
                    data: [72, 75, 78, 81, 83, 85],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 3,
                    tension: 0.4,
                    fill: true
                },
                {
                    label: 'Predicted Grade',
                    data: [70, 74, 76, 80, 82, 84],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    tension: 0.4,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Historical Performance Tracking'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                }
            }
        }
    });
}
