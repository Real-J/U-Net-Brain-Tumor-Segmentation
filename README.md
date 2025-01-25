# U-Net Brain Tumor Segmentation

This repository contains an implementation of a U-Net model for brain tumor segmentation using TensorFlow and Keras. The U-Net architecture is a popular choice for biomedical image segmentation tasks due to its encoder-decoder structure, which captures both local and global context in the input images.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Model Architecture](#model-architecture)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Introduction

This project implements a U-Net model to segment brain tumor regions from MRI scans. The model is trained on grayscale MRI images and their corresponding binary masks.

## Requirements

The following dependencies are required to run the code:

- Python 3.7+
- TensorFlow 2.0+
- NumPy
- glob
- PIL (for image loading and preprocessing)

Install the required packages using:

```bash
pip install tensorflow numpy pillow
```

## Model Architecture

The U-Net model consists of:

1. **Encoder**: A series of convolutional and max-pooling layers that extract features from the input images.
2. **Bottleneck**: The central part of the model that captures the most abstract representations.
3. **Decoder**: A series of upsampling and convolutional layers that reconstruct the segmented output.

The model uses ReLU activation in the convolutional layers and a sigmoid activation function in the output layer to produce a binary segmentation mask.

## Dataset Preparation

1. Organize the dataset with the following structure:
   ```
   dataset/
   |-- train images/
   |   |-- image1.png
   |   |-- image2.png
   |-- train mask/
   |   |-- mask1.png
   |   |-- mask2.png
   |-- test images/
   |   |-- image1.png
   |-- test mask/
   |   |-- mask1.png
   ```

2. Replace the paths in the code with the paths to your dataset.

3. Ensure all images and masks are in grayscale format and have the same resolution. Images will be resized to `(128, 128)` during preprocessing.

## Training

1. Load the dataset using the provided `load_images_and_masks` function.
2. Normalize the images and masks to the range `[0, 1]`.
3. Compile and train the U-Net model using the following command:

   ```python
   model.fit(
       x=train_images,
       y=train_masks,
       validation_data=(val_images, val_masks),
       batch_size=16,
       epochs=50
   )
   ```

4. Save the trained model using:
   ```python
   model.save('unet_brain_tumor_segmentation.h5')
   ```

## Usage

To train the model:

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/unet-brain-tumor-segmentation.git
   cd unet-brain-tumor-segmentation
   ```

2. Run the training script:
   ```bash
   python train.py
   ```

## Results

After training, the U-Net model will generate binary segmentation masks that highlight tumor regions in MRI scans. Quantitative results (e.g., accuracy and loss) will be displayed during training. You can also visualize the segmented masks to evaluate the model's performance.

## Acknowledgments

- The U-Net architecture is inspired by the paper ["U-Net: Convolutional Networks for Biomedical Image Segmentation"](https://arxiv.org/abs/1505.04597).
- Special thanks to open-source MRI datasets used for training and evaluation.

---

Feel free to contribute to this project by submitting issues or pull requests!

