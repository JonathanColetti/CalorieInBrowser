

/*
Loads both the segmented and classifer model
Then makes predictions when an image is loaded
Calculate calories using segmentation
Other utility functions and disposing
*/


/**
 * Generates a random RGB color in range
 * 
 * @returns {number[]} An array [r, g, b] representing a randomly generated color,
 *                     each channel between COLOR_MIN and COLOR_MIN + COLOR_MAX_RANDOM
 */
function generateColor() {
    const r = Math.floor(Math.random() * COLOR_MAX_RANDOM) + COLOR_MIN;
    const g = Math.floor(Math.random() * COLOR_MAX_RANDOM) + COLOR_MIN;
    const b = Math.floor(Math.random() * COLOR_MAX_RANDOM) + COLOR_MIN;
    return [r, g, b];
}



const statusText = document.getElementById('status-text');
const loader = document.getElementById('loader');
const mainContent = document.getElementById('main-content');
const imageUpload = document.getElementById('image-upload');
const uploadedImage = document.getElementById('uploaded-image');
const imagePlaceholder = document.getElementById('image-placeholder');
const resultsContent = document.getElementById('results-content');
const resultsPlaceholder = document.getElementById('results-placeholder');
const predictedFood = document.getElementById('predicted-food');
const calorieEstimate = document.getElementById('calorie-estimate');
const nerdStatsSection = document.getElementById('nerd-stats-section');
const nerdCanvas = document.getElementById('nerd-segmentation-canvas');
const allPredictionsList = document.getElementById('all-predictions-list');

let classifierModel;
let segmenterModel;

/**
 * Async loads both classifier and segmentor models in tf.js 
 * 
 * Updates UI to show progress
 * Warms up both models with dummy inputs for faster inference later + to test
 * 
 * @async
 * @returns {Promise<void>} Resolves once both models are loaded and initialized
 */
async function loadModels() {
    try {
        statusText.textContent = 'Loading classifier model...';
        classifierModel = await tf.loadLayersModel(CLASSIFIER_MODEL_PATH);

        statusText.textContent = 'Loading segmenter model...';
        segmenterModel = await tf.loadLayersModel(SEGMENTER_MODEL_PATH);

        tf.tidy(() => {
            const dummyClassifierInput = tf.zeros([1, CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE, 3]);
            const dummySegmenterInput = tf.zeros([1, SEGMENTATION_IMG_SIZE, SEGMENTATION_IMG_SIZE, 3]);
            classifierModel.predict(dummyClassifierInput);
            segmenterModel.predict(dummySegmenterInput);
        });

        statusText.textContent = 'Models Loaded! Upload an image.';
        loader.style.display = 'none';
        mainContent.classList.remove('hidden');
    } catch (error) {
        console.error("Error loading models:", error);
        statusText.textContent = 'Failed to load models.';
        loader.style.borderColor = 'red';
    }
}

/**
 * Converts an image element into a normalized TensorFlow tensor used for model input
 * 
 * @param {HTMLImageElement} imgElement - The source image element
 * @param {number} targetSize - Desired image width and height size 
 * 
 * @returns {tf.Tensor4D} A batched, normalized tensor of shape [1, targetSize, targetSize, 3] where 1 is batch and 3 is num_channels
 */
function preprocessImage(imgElement, targetSize) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imgElement);
        const resized = tf.image.resizeBilinear(tensor, [targetSize, targetSize]);
        const normalized = resized.div(NORMALIZATION_DIVISOR);
        const batched = normalized.expandDims(BATCH_DIMENSION);
        return batched;
    });
}

/**
 * Handles image upload event then runs full pipeline
 * Load uploaded image
 * Runs classification to predict food
 * Runs segmentation to find food region
 * Estimates calories based on pixel area and database values
 * Updates UI with prediction, etc
 * 
 * @async
 * @param {Event} event - Change event from file input
 * @returns {Promise<void>}
 */
async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const imageUrl = URL.createObjectURL(file);
    uploadedImage.src = imageUrl;
    uploadedImage.classList.remove('hidden');
    imagePlaceholder.classList.add('hidden');

    resultsPlaceholder.classList.add('hidden');
    resultsContent.classList.remove('hidden');
    predictedFood.textContent = 'Analyzing...';
    calorieEstimate.textContent = '...';

    nerdStatsSection.classList.add('hidden');
    allPredictionsList.innerHTML = '';

    const nerdCtx = nerdCanvas.getContext('2d');
    // x = 0, y = 0
    nerdCtx.clearRect(0, 0, nerdCanvas.width, nerdCanvas.height);

    uploadedImage.onload = async () => {
        try {
            const classificationTensor = preprocessImage(uploadedImage, CLASSIFIER_IMG_SIZE);
            const predictions = await classifierModel.predict(classificationTensor).data();
            classificationTensor.dispose();

            const allPredictions = Array.from(predictions)
                .map((probability, index) => ({
                    className: CLASS_NAMES[index],
                    classIndex: index,
                    probability
                }))
                .sort((a, b) => b.probability - a.probability);

            const topPrediction = allPredictions[0];
            const predictedClassName = topPrediction.className;
            const predictedClassIdForSegmenter = topPrediction.classIndex + 1;
            predictedFood.textContent = predictedClassName;

            const segmentationTensor = preprocessImage(uploadedImage, SEGMENTATION_IMG_SIZE);
            const outputTensor = segmenterModel.predict(segmentationTensor);
            segmentationTensor.dispose();

            const foodPixels = await countClassPixels(outputTensor, predictedClassIdForSegmenter);
            const estimatedGrams = foodPixels * PIXELS_TO_GRAMS_FACTOR;
            const caloriesPer100g = FOOD_DATABASE[predictedClassName];

            let totalCalories = 0;
            if (caloriesPer100g !== undefined) {
                totalCalories = (estimatedGrams / 100) * caloriesPer100g;
            } else {
                console.warn(`Calories per 100g not found for ${predictedClassName}. Using fallback!!.`);
                totalCalories = (estimatedGrams / 100) * DEFAULT_CALORIES_PER_100G;
            }

            calorieEstimate.textContent = Math.round(Math.max(0, totalCalories));

            await drawSegmentationMask(outputTensor, uploadedImage, SEGMENTATION_IMG_SIZE);
            renderAllPredictions(allPredictions.slice(0, TOP_K_CLASSIFIER_RESULTS));
            outputTensor.dispose();

            nerdStatsSection.classList.remove('hidden');
        } catch (error) {
            console.error("Error during analysis:", error);
            predictedFood.textContent = 'Error';
            calorieEstimate.textContent = 'N/A';
        }

        URL.revokeObjectURL(imageUrl);
    };
}

/**
 * Configuration object for all semantic classes (background + food class names)
 * Every class has a unique display color that is used in the segmentation overlay
 */
const CLASS_CONFIG = [
    { name: 'Background', color: BACKGROUND_COLOR },
    ...CLASS_NAMES.map(name => ({
        name,
        color: generateColor()
    }))
];

/**
 * Counts the number of pixels belonging to a specific food class in the segmentation
 * 
 * @async
 * @param {tf.Tensor4D} maskTensor - Output tensor from the segmentation model with shape [1, H, W, num_classes]
 * @returns {Promise<number>} The number of pixels classified as food
 */
async function countClassPixels(maskTensor) {
    const argMaxTensor = maskTensor.argMax(-1);
    const data = await argMaxTensor.data();

    let count = 0;
    for (let i = 0; i < data.length; i++) {
        if (data[i] !== BACKGROUND_CLASS_ID) count++;
    }

    argMaxTensor.dispose();
    return count;
}

/**
 * Draws a colored segmentation overlay on the canvas
 * 
 * @async
 * @param {tf.Tensor4D} maskTensor - Segmentation output tensor [1, H, W, num_classes]
 * @param {HTMLImageElement} imgElement - The original image to overlay segment mask on
 * @param {number} maskSize - The resolution (width/height) of the seg output
 * 
 * @returns {Promise<void>} Resolves when the overlay is drawn on the NERD CANVAS
 */
async function drawSegmentationMask(maskTensor, imgElement, maskSize) {
    const [height, width] = [maskSize, maskSize];
    const argMaxTensor = maskTensor.argMax(-1);
    const classMap = await argMaxTensor.data();
    argMaxTensor.dispose();

    const rgbaData = new Uint8ClampedArray(width * height * 4);

    for (let i = 0; i < classMap.length; i++) {
        const classId = classMap[i];
        const color = CLASS_CONFIG[classId].color;
        const offset = i * 4;
        rgbaData[offset] = color[0];
        rgbaData[offset + 1] = color[1];
        rgbaData[offset + 2] = color[2];
        rgbaData[offset + 3] = classId === BACKGROUND_CLASS_ID ? 0 : SEGMENT_ALPHA;
    }

    const displayWidth = imgElement.clientWidth;
    const displayHeight = imgElement.clientHeight;
    nerdCanvas.width = displayWidth;
    nerdCanvas.height = displayHeight;

    const ctx = nerdCanvas.getContext('2d');
    ctx.clearRect(0, 0, displayWidth, displayHeight);
    ctx.drawImage(imgElement, 0, 0, displayWidth, displayHeight);

    const maskImageData = new ImageData(rgbaData, width, height);
    const offscreenCanvas = document.createElement('canvas');
    offscreenCanvas.width = width;
    offscreenCanvas.height = height;
    const offscreenCtx = offscreenCanvas.getContext('2d');
    offscreenCtx.putImageData(maskImageData, 0, 0);

    ctx.globalAlpha = SEGMENT_OVERLAY_OPACITY;
    ctx.drawImage(offscreenCanvas, 0, 0, displayWidth, displayHeight);
    ctx.globalAlpha = FULL_OPACITY;
}

/**
 * Renders a list of predicted food classes with their probabilities as progress bars
 * 
 * @param {{ className: string, probability: number }[]} predictions - Array of prediction results sorted by probability
 */
function renderAllPredictions(predictions) {
    allPredictionsList.innerHTML = '';

    predictions.forEach(p => {
        const percentage = (p.probability * 100).toFixed(2);
        const predictionItem = document.createElement('div');
        predictionItem.className = 'flex flex-col space-y-1';
        predictionItem.innerHTML = `
            <div class="flex justify-between text-sm">
                <span class="font-medium text-gray-700">${p.className}</span>
                <span class="text-gray-600">${percentage}%</span>
            </div>
            <div class="progress-container">
                <div class="progress-bar" style="width: ${percentage}%;"></div>
            </div>
        `;
        allPredictionsList.appendChild(predictionItem);
    });
}

imageUpload.addEventListener('change', handleImageUpload);
loadModels();

document.querySelectorAll('details').forEach(detail => {
    detail.addEventListener('toggle', () => {
        const icon = detail.querySelector('summary svg');
        icon.style.transform = detail.open ? 'rotate(180deg)' : 'rotate(0deg)';
    });
});
