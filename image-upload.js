let uploadedImages = [];

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const imagesContainer = document.getElementById('imagesContainer');

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('highlight');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('highlight');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('highlight');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    fetchImages();
});

async function fetchImages() {
    try {
        const response = await fetch('http://localhost:5000/images');
        if (!response.ok) throw new Error(`Server error: ${response.status}`);

        const images = await response.json();
        console.log("Fetched images:", images); // Debugging log

        images.forEach(image => {
            const imageData = {
                id: image._id,
                name: image.filename,
                src: `http://localhost:5000${image.path}`, 
                date: new Date(image.uploadDate).toLocaleDateString()
            };
            console.log("Rendering image:", imageData); // Debugging log

            uploadedImages.push(imageData);
            renderImage(imageData);
        });
    } catch (error) {
        console.error('Error fetching images:', error);
    }
}

function handleFiles(files) {
    Array.from(files).forEach(file => {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();

            reader.onload = (e) => {
                const image = {
                    id: Date.now(),
                    name: file.name,
                    src: e.target.result,
                    file: file
                };

                uploadedImages.push(image);
                renderImage(image);
                uploadImageToServer(image);
            };

            reader.readAsDataURL(file);
        } else {
            alert('Only image files are allowed!');
        }
    });
}

function renderImage(image) {
    const container = document.getElementById('imagesContainer');
    const imageElement = document.createElement('div');
    imageElement.className = 'image-item';
    imageElement.dataset.id = image.id;

    imageElement.innerHTML = `
        <div class="image-preview">
            <img src="${image.src}" alt="${image.name}">
        </div>
        <div class="image-info">
            <h4>${image.name}</h4>
            <p>${image.date}</p>
            <button class="describe-btn" onclick="createDescription('${image.id}')">
                <i class="fas fa-comment-alt"></i> Create Description
            </button>
            <div class="image-description" style="display: none;">
                <h4>Description:</h4>
                <p class="description-text">Not analyzed yet.</p>
            </div>
        </div>
    `;

    container.insertBefore(imageElement, container.firstChild);
}

function uploadImageToServer(image) {
    const formData = new FormData();
    formData.append('image', image.file);

    fetch('http://localhost:5000/upload_image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log("Image uploaded successfully:", data);
        } else {
            console.error("Error uploading image:", data.error);
        }
    })
    .catch(error => console.error("Server error:", error));
}


function createDescription(imageId) {
    const imageElement = document.querySelector(`.image-item[data-id="${imageId}"]`);
    if (!imageElement) return;

    const descriptionDiv = imageElement.querySelector('.image-description');
    const descriptionText = imageElement.querySelector('.description-text');
    const loadingIcon = document.createElement("span");

    // Show loading state
    descriptionDiv.style.display = "block";
    descriptionText.textContent = "Analyzing...";
    loadingIcon.className = "loading-icon"; // Add a CSS spinner
    descriptionDiv.appendChild(loadingIcon);

    fetch("http://localhost:5000/analyze_image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageId })
    })
    .then(response => response.json())
    .then(data => {
        loadingIcon.remove(); // Remove loading icon

        if (data.success) {
            descriptionText.textContent = data.description;
        } else {
            descriptionText.textContent = "Error: " + (data.error || "Unknown issue");
        }
    })
    .catch(error => {
        loadingIcon.remove();
        descriptionText.textContent = "Error analyzing image.";
        console.error("Error:", error);
    });
}