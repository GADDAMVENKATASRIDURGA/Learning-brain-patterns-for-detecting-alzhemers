# Import required libraries
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set image size
IMG_SIZE = 128

# Load pretrained VGG16 model (feature extractor)
base_model = VGG16(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE,IMG_SIZE,3))
base_model.trainable = False

# Build model using transfer learning
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load and preprocess dataset
train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory("Datasets/train",
                                           target_size=(IMG_SIZE,IMG_SIZE),
                                           class_mode='categorical')
test_data = test_gen.flow_from_directory("Datasets/test",
                                         target_size=(IMG_SIZE,IMG_SIZE),
                                         class_mode='categorical')

# Train the model
model.fit(train_data, epochs=15, validation_data=test_data)

# Save the trained model
model.save("model/vgg16.h5")

# Display completion message
print("✅ VGG16 model saved")
