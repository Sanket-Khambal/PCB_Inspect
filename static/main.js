// PCB Defect Detection - Main JavaScript

// Configuration Sliders
const confSlider = document.getElementById('confSlider');
const confValue = document.getElementById('confValue');
const iouSlider = document.getElementById('iouSlider');
const iouValue = document.getElementById('iouValue');

confSlider.addEventListener('input', (e) => {
    confValue.textContent = parseFloat(e.target.value).toFixed(2);
});

iouSlider.addEventListener('input', (e) => {
    iouValue.textContent = parseFloat(e.target.value).toFixed(2);
});

// File Upload Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingDiv = document.getElementById('loadingDiv');
const emptyState = document.getElementById('emptyState');
const resultsContainer = document.getElementById('resultsContainer');

let selectedFile = null;

// Upload Area Click Handler
uploadArea.addEventListener('click', () => fileInput.click());

// Drag and Drop Handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// File Input Change Handler
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Handle File Upload
function handleFile(file) {
    // Validate file type
    if (!file.type.match('image/(jpeg|jpg|png)')) {
        alert('Please upload a JPG, JPEG, or PNG image');
        return;
    }

    selectedFile = file;
    
    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
        analyzeBtn.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

// Analyze Button Click Handler
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    // Show loading, hide other states
    emptyState.style.display = 'none';
    resultsContainer.style.display = 'none';
    loadingDiv.style.display = 'block';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            emptyState.style.display = 'flex';
            return;
        }

        displayResults(data);
    } catch (error) {
        alert('Error: ' + error.message);
        emptyState.style.display = 'flex';
    } finally {
        loadingDiv.style.display = 'none';
    }
});

// Display Results Function
function displayResults(data) {
    resultsContainer.style.display = 'block';

    // Update Verdict Badge
    const verdictBadge = document.getElementById('verdictBadge');
    verdictBadge.textContent = data.verdict;
    verdictBadge.className = `verdict-badge verdict-${data.verdict.toLowerCase()}`;

    // Update Metrics
    document.getElementById('totalDefects').textContent = data.num_detections;
    
    const criticalCount = data.detections.filter(d => d.severity === 'CRITICAL').length;
    const warningCount = data.detections.filter(d => d.severity === 'WARNING').length;
    
    document.getElementById('criticalDefects').textContent = criticalCount;
    document.getElementById('warningDefects').textContent = warningCount;

    // Display Annotated Image
    document.getElementById('annotatedImage').src = `/results/${data.annotated_filename}`;

    // Generate Defect List
    const defectList = document.getElementById('defectListContent');
    defectList.innerHTML = '';

    if (data.detections.length === 0) {
        defectList.innerHTML = '<div style="text-align: center; color: #30d158; padding: 12px;">✓ No defects detected</div>';
    } else {
        data.detections.forEach((defect, index) => {
            const item = document.createElement('div');
            item.className = `defect-item ${defect.severity.toLowerCase()}`;
            
            const confidence = (defect.confidence * 100).toFixed(1);
            
            item.innerHTML = `
                <div class="defect-header">
                    <span class="defect-name">${index + 1}. ${defect.class}</span>
                    <span class="defect-confidence">${confidence}%</span>
                </div>
                <div class="defect-details">
                    ${defect.severity} • 
                    (${Math.round(defect.bbox.x1)}, ${Math.round(defect.bbox.y1)}) • 
                    ${Math.round(defect.bbox.width)}×${Math.round(defect.bbox.height)}px
                </div>
            `;
            
            defectList.appendChild(item);
        });
    }
}