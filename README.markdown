# In-Demand Machine Learning Models for 2025

This repository contains three machine learning models that are highly demanded in 2025: an XGBoost classifier for tabular data, a ResNet-50 image classifier for computer vision, and a BERT-based text classifier for NLP.

## Models Included

1. **XGBoost Classifier** (`xgboost_classifier.py`): Classifies tennis swing types using tabular data.
2. **ResNet-50 Image Classifier** (`resnet_classifier.py`): Classifies images using a pre-trained ResNet-50 model.
3. **BERT Text Classifier** (`bert_classifier.py`): Performs sentiment analysis on text using BERT.

## Prerequisites

- Python 3.8+

- Install dependencies:

  ```bash
  pip install pandas xgboost scikit-learn joblib torch torchvision transformers pillow numpy
  ```

## Dataset and Files

- **swing_data.csv**: Required for `xgboost_classifier.py`. Format: columns `accelX`, `accelY`, `accelZ`, `gyroX`, `gyroY`, `gyroZ`, `swingType`.
- **example_image.jpg**: Required for `resnet_classifier.py`. Replace with your image.
- **imagenet_classes.txt**: Required for `resnet_classifier.py`. Download from ImageNet or similar sources.

## Usage

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Run each script:

   - XGBoost: `python xgboost_classifier.py`
   - ResNet: `python resnet_classifier.py`
   - BERT: `python bert_classifier.py`

## Notes

- Ensure you have a GPU for faster inference with ResNet and BERT models.
- Replace placeholder files (e.g., `example_image.jpg`) with your data.

## License

MIT License