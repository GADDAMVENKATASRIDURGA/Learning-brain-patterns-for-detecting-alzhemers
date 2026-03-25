import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 128

base = MobileNet(weights="imagenet", include_top=False,
                 input_shape=(IMG_SIZE,IMG_SIZE,3))
base.trainable = False

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

gen = ImageDataGenerator(rescale=1./255)

train = gen.flow_from_directory("dataset/train",
                                target_size=(IMG_SIZE,IMG_SIZE),
                                class_mode='categorical')
test = gen.flow_from_directory("dataset/test",
                               target_size=(IMG_SIZE,IMG_SIZE),
                               class_mode='categorical')

model.fit(train, epochs=15, validation_data=test)
model.save("model/mobilenet.h5")

print("✅ MobileNet model saved")
