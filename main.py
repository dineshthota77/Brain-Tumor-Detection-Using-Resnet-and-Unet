import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Flatten, Input,
                                     GlobalAveragePooling2D, UpSampling2D, Concatenate,
                                     BatchNormalization, Dropout)

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from google.colab import drive

# Mount Google Drive to save the model
drive.mount('/content/drive')

# Image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 32

# Directories for the dataset
train_dir = "/content/drive/MyDrive/b2/Brain/train"
val_dir = "/content/drive/MyDrive/b2/Brain/validation"
test_dir = "/content/drive/MyDrive/b2/Brain/test"

# Data generators with extensive augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# UNet model for segmentation
def unet_model(input_shape):
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)

    # Decoder
    u1 = UpSampling2D((2, 2))(c3)
    u1 = Concatenate()([u1, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = BatchNormalization()(c4)

    u2 = UpSampling2D((2, 2))(c4)
    u2 = Concatenate()([u2, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.5)(c5)

    # Segmentation output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c5)

    return Model(inputs, outputs)

# ResNet for feature extraction
def build_feature_extraction_resnet(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze some layers to focus on classification layers first
    for layer in base_model.layers[:100]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    return Model(inputs=base_model.input, outputs=x)

# Classification CNN
def build_classification_cnn(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.5),
        Dense(4, activation='softmax')  # 4 classes
    ])
    return model

# Integrating UNet and ResNet
def full_pipeline(input_shape):
    unet = unet_model(input_shape)
    resnet = build_feature_extraction_resnet((img_height, img_width, 3))
    classifier = build_classification_cnn((2048,))

    inputs = Input(input_shape)
    segmented = unet(inputs)
    converted = Conv2D(3, (1, 1), activation='relu')(segmented)
    features = resnet(converted)
    classified = classifier(features)

    final_model = Model(inputs=inputs, outputs=classified)
    return final_model

# Compile the model
model = full_pipeline((img_height, img_width, 3))
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for training
model_checkpoint = ModelCheckpoint('/content/drive/MyDrive/best_model.keras', monitor='val_accuracy', save_best_only=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

# Train the model for 40 epochs
history = model.fit(train_generator, validation_data=val_generator, epochs=55, callbacks=[model_checkpoint, lr_scheduler])

# Save the model to Google Drive
model.save('/content/drive/MyDrive/best_model_final.keras')

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {accuracy:.2f}')

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'bo-', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'bo-', label='Training Loss')
plt.plot(history.history['val_loss'], 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
