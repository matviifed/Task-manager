// Store for uploaded images
let uploadedImages = [];

// Setup event listeners
document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const viewButtons = document.querySelectorAll('.view-btn');
    const modal = document.getElementById('imageModal');
    const closeBtn = modal.querySelector('.close-btn');

    // Drag and drop handlers
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
        handleFiles(e.dataTransfer.files);
    });

    // File input handler
    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    // View toggle handlers
    viewButtons.forEach(button => {
        button.addEventListener('click', () => {
            const view = button.dataset.view;
            toggleView(view);
            viewButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
        });
    });

    // Modal handlers
    closeBtn.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.display = 'none';
        }
    });
});

// Handle uploaded files
function handleFiles(files) {
    Array.from(files).forEach(file => {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                const image = {
                    id: Date.now(),
                    name: file.name,
                    size: formatSize(file.size),
                    date: new Date().toLocaleDateString(),
                    src: e.target.result
                };
                
                uploadedImages.push(image);
                renderImage(image);
            };
            
            reader.readAsDataURL(file);
        }
    });
}

// Render image card
function renderImage(image) {
    const container = document.getElementById('imagesContainer');
    const card = document.createElement('div');
    card.className = 'image-card';
    card.onclick = () => showImageDetails(image);
    
    card.innerHTML = `
        <img src="${image.src}" alt="${image.name}">
        <div class="image-info">
            <h4>${image.name}</h4>
            <p>${image.size} • ${image.date}</p>
        </div>
    `;
    
    container.insertBefore(card, container.firstChild);
}

// Show image details in modal
function showImageDetails(image) {
    const modal = document.getElementById('imageModal');
    const modalImage = document.getElementById('modalImage');
    const imageName = document.getElementById('imageName');
    const imageSize = document.getElementById('imageSize');
    const uploadDate = document.getElementById('uploadDate');
    
    modalImage.src = image.src;
    imageName.textContent = image.name;
    imageSize.textContent = `Size: ${image.size}`;
    uploadDate.textContent = `Uploaded: ${image.date}`;
    
    modal.style.display = 'block';
}

// Toggle view (grid/list)
function toggleView(view) {
    const container = document.getElementById('imagesContainer');
    container.className = `images-container ${view}-view`;
}

// Format file size
function formatSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}