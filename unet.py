import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array
import glob

# Define U-Net model
def unet(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    d = Dropout(0.5)(c4)
    p4 = MaxPooling2D((2, 2))(d)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs, outputs)
    return model

# Compile the model
model = unet()
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Custom data loader
def load_images_and_masks(image_dir, mask_dir, target_size=(128, 128)):
    image_files = sorted(glob.glob(f"{image_dir}/*.png"))  # Adjust file extension as needed
    mask_files = sorted(glob.glob(f"{mask_dir}/*.png"))

    images = [img_to_array(load_img(f, target_size=target_size, color_mode='grayscale')) for f in image_files]
    masks = [img_to_array(load_img(f, target_size=target_size, color_mode='grayscale')) for f in mask_files]

    return np.array(images), np.array(masks)

# Paths to your data
train_images_path = "L:/brain data/train images"
train_masks_path = "L:/brain data/train mask"
val_images_path = "L:/brain data/test images"
val_masks_path = "L:/brain data/test mask"

# Load data
train_images, train_masks = load_images_and_masks(train_images_path, train_masks_path)
val_images, val_masks = load_images_and_masks(val_images_path, val_masks_path)

# Normalize data
train_images = train_images / 255.0
train_masks = train_masks / 255.0
val_images = val_images / 255.0
val_masks = val_masks / 255.0

# Train the model
model.fit(
    x=train_images,
    y=train_masks,
    validation_data=(val_images, val_masks),
    batch_size=16,
    epochs=50
)

# Save the model
model.save('unet_brain_tumor_segmentation.h5')
