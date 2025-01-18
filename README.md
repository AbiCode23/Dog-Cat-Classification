

# Dog and Cat Classification using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to classify images as either dogs or cats. The model is built using TensorFlow and Keras, and is deployed through a user-friendly Streamlit interface. The CNN achieves high accuracy in distinguishing between dogs and cats in images through deep learning techniques.

## Features
- Deep learning model built with TensorFlow and Keras
- Interactive web interface using Streamlit
- Real-time image classification
- Confidence score display
- Support for various image formats (JPG, JPEG, PNG)
- Easy-to-use upload interface

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Streamlit
- PIL (Python Imaging Library)
- NumPy
- OpenCV
- Matplotlib
- Kaggle

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Dog-Cat-Classification.git
cd Dog-Cat-Classification
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset
The model is trained on the "Dogs vs Cats" dataset from Kaggle. To download the dataset:

1. Install the Kaggle CLI:
```bash
pip install kaggle
```

2. Configure Kaggle credentials:
- Download `kaggle.json` from your Kaggle account
- Place it in `~/.kaggle/` directory
- Set appropriate permissions:
```bash
chmod 600 ~/.kaggle/kaggle.json
```

3. Download the dataset:
```bash
kaggle datasets download salader/dogs-vs-cats
unzip dogs-vs-cats.zip
```

## Model Architecture
The CNN model consists of:
- Input Layer (256x256x3)
- Three Convolutional Blocks, each containing:
  - Conv2D Layer
  - BatchNormalization
  - MaxPooling2D
- Flatten Layer
- Two Dense Layers with Dropout
- Output Layer (Sigmoid activation)

## Training the Model
To train the model:

1. Ensure the dataset is properly organized:
```
data/
├── train/
│   ├── dogs/
│   └── cats/
└── test/
    ├── dogs/
    └── cats/
```

2. Run the training notebook:
```bash
jupyter notebook notebooks/dog_cat_classification.ipynb
```

3. Follow the notebook instructions to train and save the model

## Running the Application

1. Ensure the trained model is in the correct location:
```bash
models/dog_cat_classifier.keras
```

2. Start the Streamlit application:
```bash
cd app
streamlit run streamlit_app.py
```

3. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

## Usage
1. Launch the Streamlit app
2. Use the sidebar to configure the model path if necessary
3. Upload an image using the file uploader
4. Click "Submit" to get the classification result
5. View the prediction and confidence score

## Model Performance
The model achieves:
- Training Accuracy: ~90%
- Validation Accuracy: ~85%
- Binary Cross-Entropy Loss: ~0.3

## File Structure
```
Dog-Cat-Classification/
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── dog_cat_classification.ipynb
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## Troubleshooting
Common issues and solutions:

1. Model loading error:
   - Verify the model path in the Streamlit app
   - Ensure the model file exists and is accessible

2. Image processing error:
   - Check if the image format is supported (JPG, JPEG, PNG)
   - Verify the image isn't corrupted

3. Memory issues:
   - Reduce the batch size in the application
   - Close other resource-intensive applications

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset provided by Kaggle
- TensorFlow and Keras teams
- Streamlit community

## Contact
For questions or feedback, please open an issue in the GitHub repository.

---
Made with ❤️ by [Your Name]
