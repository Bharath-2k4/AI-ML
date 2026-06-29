// ===============================
// GLOBAL INITIALIZATION
// ===============================
document.addEventListener('DOMContentLoaded', function () {

    // -------------------------------
    // Bootstrap tooltips (safe)
    // -------------------------------
    if (typeof bootstrap !== "undefined") {
        const tooltipTriggerList = [].slice.call(
            document.querySelectorAll('[data-bs-toggle="tooltip"]')
        );
        tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));
    }

    // -------------------------------
    // Fade-in animation
    // -------------------------------
    document.querySelectorAll('[data-animate]').forEach(el => {
        el.classList.add('fade-in');
    });

    // -------------------------------
    // Initialize confidence bars safely
    // -------------------------------
    updateConfidenceBars();
});


// ===============================
// AJAX UTILITY
// ===============================
function makeRequest(url, method = 'GET', data = null) {
    return fetch(url, {
        method: method,
        headers: {
            'Content-Type': 'application/json'
        },
        body: data ? JSON.stringify(data) : null
    })
        .then(res => res.json())
        .catch(err => {
            console.error('Request failed:', err);
            return { error: 'Request failed' };
        });
}


// ===============================
// LOADING BUTTON HELPERS
// ===============================
function showLoading(element) {
    if (!element) return '';
    element.disabled = true;
    const originalText = element.innerHTML;
    element.innerHTML = '<span class="loading-spinner me-2"></span>Processing...';
    return originalText;
}

function hideLoading(element, originalText) {
    if (!element) return;
    element.disabled = false;
    element.innerHTML = originalText;
}


// ===============================
// ALERT SYSTEM
// ===============================
function showAlert(message, type = 'info') {
    const container = document.querySelector('.container');
    if (!container) return;

    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    container.insertBefore(alertDiv, container.firstChild);

    setTimeout(() => {
        if (alertDiv.parentElement) alertDiv.remove();
    }, 5000);
}


// ===============================
// CONFIDENCE BAR (SAFE VERSION)
// ===============================
function updateConfidenceBars() {
    const bars = document.querySelectorAll('.confidence-bar');

    if (!bars || bars.length === 0) return;

    bars.forEach(bar => {
        const fill = bar.querySelector('.confidence-fill');

        // 🔐 HARD GUARD (THIS LINE SAVES YOU)
        if (!fill) return;

        const confidenceAttr = fill.getAttribute('data-confidence');
        if (!confidenceAttr) return;

        const confidence = parseFloat(confidenceAttr);
        if (isNaN(confidence)) return;

        fill.style.width = `${confidence}%`;

        fill.classList.remove(
            'confidence-high',
            'confidence-medium',
            'confidence-low'
        );

        if (confidence >= 80) {
            fill.classList.add('confidence-high');
        } else if (confidence >= 60) {
            fill.classList.add('confidence-medium');
        } else {
            fill.classList.add('confidence-low');
        }
    });
}



// ===============================
// DATE FORMATTER
// ===============================
function formatDate(dateString) {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleString();
}


// ===============================
// FEATURE TEST BUTTON
// ===============================
const featureBtn = document.getElementById('generateFeatureSample');

if (featureBtn) {
    featureBtn.addEventListener('click', function () {
        const originalText = showLoading(featureBtn);

        makeRequest('/generate_feature_sample')
            .then(data => {
                hideLoading(featureBtn, originalText);

                if (data.error) {
                    showAlert(data.error, 'danger');
                    return;
                }

                // SAFE SETTERS
                const safeText = (id, val) => {
                    const el = document.getElementById(id);
                    if (el) el.textContent = val;
                };

                safeText('sampleIndex', data.sample_index);
                safeText('actualCheat', data.actual_cheat ? 'Cheater' : 'Legit');
                safeText('actualChurn', data.actual_churn ? 'Likely to Leave' : 'Will Stay');
                safeText('predictedCheat', data.predicted_cheat ? 'Cheater' : 'Legit');
                safeText('predictedChurn', data.predicted_churn ? 'Likely to Leave' : 'Will Stay');
                safeText('cheatConfidence', `${data.cheat_confidence}%`);
                safeText('churnConfidence', `${data.churn_confidence}%`);

                // Confidence bars
                const cheatBar = document.getElementById('cheatConfidenceBar');
                const churnBar = document.getElementById('churnConfidenceBar');

                if (cheatBar) cheatBar.setAttribute('data-confidence', data.cheat_confidence);
                if (churnBar) churnBar.setAttribute('data-confidence', data.churn_confidence);

                // Reasons
                const cheatReasonsList = document.getElementById('cheatReasons');
                const churnReasonsList = document.getElementById('churnReasons');

                if (cheatReasonsList) {
                    cheatReasonsList.innerHTML = '';
                    (data.cheat_reasons || []).forEach(reason => {
                        const li = document.createElement('li');
                        li.className = 'list-group-item';
                        li.textContent = reason;
                        cheatReasonsList.appendChild(li);
                    });
                }

                if (churnReasonsList) {
                    churnReasonsList.innerHTML = '';
                    (data.churn_reasons || []).forEach(reason => {
                        const li = document.createElement('li');
                        li.className = 'list-group-item';
                        li.textContent = reason;
                        churnReasonsList.appendChild(li);
                    });
                }

                // Sample data
                const sampleDataDiv = document.getElementById('sampleData');
                if (sampleDataDiv) {
                    sampleDataDiv.innerHTML = '';
                    Object.entries(data.sample_data || {}).forEach(([k, v]) => {
                        const row = document.createElement('div');
                        row.className = 'row mb-2';
                        row.innerHTML = `
                            <div class="col-6"><strong>${k}:</strong></div>
                            <div class="col-6">${v}</div>
                        `;
                        sampleDataDiv.appendChild(row);
                    });
                }

                // Show results
                const resultsSection = document.getElementById('resultsSection');
                if (resultsSection) {
                    resultsSection.classList.remove('d-none');
                    resultsSection.classList.add('slide-up');
                }

                updateConfidenceBars();
                showAlert('Sample generated successfully!', 'success');
            });
    });
}


// ===============================
// LOG TEST FORM
// ===============================
const logForm = document.getElementById('logTestForm');

if (logForm) {
    logForm.addEventListener('submit', function (e) {
        e.preventDefault();

        const btn = logForm.querySelector('button[type="submit"]');
        const originalText = showLoading(btn);

        const formData = new FormData(logForm);
        const payload = Object.fromEntries(formData);

        makeRequest('/generate_log_prediction', 'POST', payload)
            .then(data => {
                hideLoading(btn, originalText);

                if (data.error) {
                    showAlert(data.error, 'danger');
                    return;
                }

                showAlert('Log analysis completed!', 'success');
                window.location.href = '/results';
            });
    });
}
