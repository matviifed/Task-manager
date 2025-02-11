// Slider functionality
let currentSlide = 0;
const slider = document.getElementById('goalsSlider');
const slides = Array.from(document.querySelectorAll('.slide'));
const dotsContainer = document.getElementById('sliderDots');
let slidesPerView = getSlidesPerView();
let autoSlideInterval;
let isTransitioning = false;

function initSlider() {
    cloneSlides();
    slides.forEach((_, index) => {
        const dot = document.createElement('div');
        dot.className = `dot ${index === 0 ? 'active' : ''}`;
        dot.addEventListener('click', () => goToSlide(index));
        dotsContainer.appendChild(dot);
    });
    window.addEventListener('resize', handleResize);
    
    updateSliderView();
    startAutoSlide();
    addDragEvents();
}

function getSlidesPerView() {
    if (window.innerWidth < 768) return 1;
    if (window.innerWidth < 1024) return 2;
    return 3;
}

function handleResize() {
    const newSlidesPerView = getSlidesPerView();
    if (newSlidesPerView !== slidesPerView) {
        slidesPerView = newSlidesPerView;
        currentSlide = slidesPerView;
        updateSliderView();
    }
}

function cloneSlides() {
    const sliderContainer = slider.parentElement;
    const firstClones = slides.slice(0, slidesPerView).map(slide => {
        let clone = slide.cloneNode(true);
        clone.classList.add('clone');
        return clone;
    });
    const lastClones = slides.slice(-slidesPerView).map(slide => {
        let clone = slide.cloneNode(true);
        clone.classList.add('clone');
        return clone;
    });
    firstClones.forEach(clone => slider.appendChild(clone));
    lastClones.forEach(clone => slider.insertBefore(clone, slider.firstChild));
    currentSlide = slidesPerView;
}

function moveSlider(direction) {
    if (isTransitioning) return;
    isTransitioning = true;
    currentSlide += direction;
    updateSliderView();
    
    slider.addEventListener('transitionend', () => {
        if (currentSlide >= slides.length + slidesPerView) {
            slider.style.transition = 'none';
            currentSlide = slidesPerView;
            updateSliderView(false);
        } else if (currentSlide < slidesPerView) {
            slider.style.transition = 'none';
            currentSlide = slides.length + slidesPerView - 1;
            updateSliderView(false);
        }
        isTransitioning = false;
    }, { once: true });
}

function goToSlide(index) {
    currentSlide = index + slidesPerView;
    updateSliderView();
    restartAutoSlide();
}

function updateSliderView(animate = true) {
    const slideWidth = 100 / slidesPerView;
    if (animate) slider.style.transition = 'transform 0.3s ease-in-out';
    else slider.style.transition = 'none';
    slider.style.transform = `translateX(-${currentSlide * slideWidth}%)`;
    
    const dots = document.querySelectorAll('.dot');
    dots.forEach((dot, index) => {
        dot.classList.toggle('active', index === (currentSlide - slidesPerView) % slides.length);
    });
}

function startAutoSlide() {
    autoSlideInterval = setInterval(() => {
        moveSlider(1);
    }, 5000);
}

function restartAutoSlide() {
    clearInterval(autoSlideInterval);
    startAutoSlide();
}

function addDragEvents() {
    let startX;
    let isDragging = false;
    
    slider.style.cursor = 'grab';
    
    slider.addEventListener('mousedown', (event) => {
        isDragging = true;
        startX = event.clientX;
        slider.style.cursor = 'grabbing';
    });
    
    slider.addEventListener('mousemove', (event) => {
        if (!isDragging) return;
        let moveX = event.clientX - startX;
        if (moveX > 50) {
            moveSlider(-1);
            isDragging = false;
        } else if (moveX < -50) {
            moveSlider(1);
            isDragging = false;
        }
    });
    
    slider.addEventListener('mouseup', () => {
        isDragging = false;
        slider.style.cursor = 'grab';
    });
    
    slider.addEventListener('mouseleave', () => {
        isDragging = false;
        slider.style.cursor = 'grab';
    });
}

document.addEventListener('DOMContentLoaded', initSlider);