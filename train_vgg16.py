import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128

base_model = VGG16(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE,IMG_SIZE,3))
base_model.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory("dataset/train",
                                           target_size=(IMG_SIZE,IMG_SIZE),
                                           class_mode='categorical')
test_data = test_gen.flow_from_directory("dataset/test",
                                         target_size=(IMG_SIZE,IMG_SIZE),
                                         class_mode='categorical')

model.fit(train_data, epochs=15, validation_data=test_data)
model.save("model/vgg16.h5")

print("✅ VGG16 model saved")
