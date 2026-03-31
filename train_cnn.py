# Import required libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set hyperparameters (image size, batch size, epochs)
IMG_SIZE = 128
BATCH = 16
EPOCHS = 20

# Define dataset paths
train_dir = "DataSets/train"
test_dir = "Datasets/test"

# Create data generators with normalization and validation split
train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_gen = ImageDataGenerator(rescale=1./255)

# Load training and validation data
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset='training'
)

test_data = train_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset='validation'
)

# Build CNN model architecture
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(IMG_SIZE,IMG_SIZE,3)),
    MaxPooling2D(),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(4,activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(train_data, epochs=EPOCHS, validation_data=test_data)

# Save the trained model
model.save("model/cnn.h5")

# Display completion message
print("✅ CNN model saved")
