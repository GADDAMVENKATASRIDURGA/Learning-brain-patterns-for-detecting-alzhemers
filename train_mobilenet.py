# Import required libraries
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set image size
IMG_SIZE = 128

# Load pretrained MobileNet model (feature extractor)
base = MobileNet(weights="imagenet", include_top=False,
                 input_shape=(IMG_SIZE,IMG_SIZE,3))
base.trainable = False

# Build model using transfer learning
model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load and preprocess dataset
gen = ImageDataGenerator(rescale=1./255)

train = gen.flow_from_directory("Datasets/train",
                                target_size=(IMG_SIZE,IMG_SIZE),
                                class_mode='categorical')
test = gen.flow_from_directory("Datasets/test",
                               target_size=(IMG_SIZE,IMG_SIZE),
                               class_mode='categorical')

# Train the model
model.fit(train, epochs=15, validation_data=test)

# Save the trained model
model.save("model/mobilenet.h5")

# Display completion message
print("✅ MobileNet model saved")
