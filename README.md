
# Calorie counter in browser
Web-based application that uses deep learning models directly in the browser to identify food from an image (classifier), estimate its volume (segmentation), and provide a rough calorie count. This project demonstrates the power of running AI tasks entirely on the client side using TensorFlow.js.

# URL [`Webapp`]()

# IMPORTANT INFO
Still in early stages (needs training on larger epochs, better calorie count, better food db, better/newer arch) but functional.


# Stack
- Frontend = HTML, CSS, JavaScript, Tailwind
- Machine Learning Frameworks = TensorFlow, Keras, Tensorflow.js
- Notebooks used to train and convert the model

# Dataset
- Food101 - Classification - [`Food101`](https://www.tensorflow.org/datasets/catalog/food101)
- Food103 - Segmentation - [`EduardoPacheco/FoodSeg103`](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103)


# Images
![Example Prediction] (./example_usage.jpg)

![Home page] (./start_page.jpg)

# Challenges

## Dataset

# TODO
- Train higher epochs
- Optimizing food db