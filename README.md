# Histopathological Cancer Detection Using Deep Learning

## Project Overview

Histopathological Cancer Detection is a deep learning-based approach to classifying cancerous and non-cancerous tissue from histopathological images. By leveraging state-of-the-art techniques in image classification, this project aims to improve diagnostic accuracy and assist medical professionals in detecting cancer more efficiently. The model employs **EfficientNetB3** as its base architecture, fine-tuned on histopathological images for binary classification.

This project was developed solely by **[Swapnil Verma](https://github.com/Swapnil-Verma24)**, covering every aspect from data preprocessing to model training, evaluation, and visualization.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Contributions](#contributions)
- [Future Work](#future-work)
- [License](#license)

## Dataset

- **Source**: The dataset used for this project comes from the [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection) competition on Kaggle.
- **Data**: The dataset consists of labeled 96x96 pixel images in `.tif` format, where each image is classified as either cancerous (`1`) or non-cancerous (`0`).
- **Classes**: Binary classification (cancerous vs. non-cancerous).
- **Data Augmentation**: To improve model robustness, several augmentation techniques such as rotation, zoom, shear, and horizontal flips are applied.

### Label Distribution

The label distribution is imbalanced, with more non-cancerous images than cancerous. To account for this, stratified sampling was used during the train-validation split.

## Installation

To get started with this project, you’ll need to have Python 3.x and several libraries installed. Follow the instructions below to set up your environment:

### Clone the Repository

```bash
git clone https://github.com/your-repo/histopathological-cancer-detection.git
cd histopathological-cancer-detection
```
### Install Dependencies
Install all required Python packages using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```
## Model Architecture
The model is based on EfficientNetB3, a pre-trained convolutional neural network known for its efficiency and strong performance on image classification tasks. The following layers were added to fine-tune the model for histopathological cancer detection:

- **GlobalAveragePooling2D**: Reduces the spatial dimensions of the feature maps.
- **Dense Layer**: A fully connected layer with ReLU activation for high-level feature extraction.
- **Dropout Layer**: Applied to prevent overfitting.
- **BatchNormalization**: Introduced for improved convergence and model generalization.
- **Sigmoid Output Layer**: Produces binary classification predictions.
### Fine-tuning
After freezing the base EfficientNetB3 model, additional layers were trained to adapt the model to the specific problem of histopathological cancer detection. Dropout and batch normalization were added to improve generalization.

## Training and Evaluation
The model was trained using the binary crossentropy loss function and the Adam optimizer with default settings. The following techniques were used for training and evaluation:

- **Early Stopping**: Stops training if validation loss does not improve after 5 epochs.
- **ReduceLROnPlateau**: Reduces learning rate if validation accuracy plateaus.
- **ImageDataGenerator**: Used for real-time data augmentation during training.
### Training Process
1. **Data Preprocessing**: The dataset was preprocessed by normalizing the pixel values to a [0, 1] range and applying several augmentation techniques like rotation, zoom, and flipping to diversify the training data.
2. **Train-Validation Split**: The dataset was split into 80% training data and 20% validation data using stratified sampling to maintain class balance.
3. **Model Training**: The model was trained for 20 epochs, with an early stopping mechanism to avoid overfitting.
4. **Evaluation**: Precision, recall, and accuracy metrics were tracked on the validation set.
### Performance Metrics
- **Accuracy**: 80%+
- **Precision**: Focused on improving precision to minimize false positives, a key metric in cancer detection tasks.
- **Recall**: Evaluated alongside precision to ensure that the model is not missing too many positive cases.
Precision-Recall curves were plotted to provide a deeper understanding of model performance across varying thresholds.

## Results
The model achieved strong performance, with the following results on the validation set:

- **Validation Accuracy**: 0.8040
- **Precision**: 0.8125
- **Recall**: 0.7921
These metrics indicate that the model is capable of effectively classifying cancerous and non-cancerous tissue, with a good balance between precision and recall.

## Contributions
This project was developed solely by [Swapnil Verma](https://github.com/Swapnil-Verma24), with contributions across the following areas:

- **Data Preprocessing**: Loading and augmenting images, splitting data, and handling imbalanced classes.
- **Model Design**: Implementing and fine-tuning the EfficientNetB3 architecture for histopathological cancer detection.
- **Training and Evaluation**: Applying appropriate loss functions, optimizers, and callbacks to ensure smooth training.
- **Visualization**: Implementing visualizations for loss, accuracy, and precision-recall curves to evaluate the model's performance.
## Future Work
- **Model Optimization**: Further hyperparameter tuning, including exploring different dropout rates, batch sizes, and learning rates to optimize performance.
- **Explainability**: Integrate explainable AI techniques like Grad-CAM to identify the regions of the images that are most important for classification decisions.
- **Segmentation**: Extend the model to not only classify images but also segment regions of interest within the tissue samples.
- **Deployment**: Deploy the model to a cloud-based or on-premise solution for real-time inference in clinical settings.
- **Larger Input Sizes**: Experiment with higher resolution images (e.g., 224x224) to capture more fine-grained details in the histopathological images.
## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/Swapnil-Verma24/Histopathalogic-Cancer-Detection/blob/main/LICENSE) file for more information.
