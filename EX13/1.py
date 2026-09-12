import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from PIL import Image

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train_norm = X_train / 255.0
X_test_norm = X_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_norm, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

def predict_custom_image(image_path, model):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    img_norm = img_array / 255.0
    input_tensor = np.expand_dims(img_norm, axis=0)
    predictions = model.predict(input_tensor)
    predicted_digit = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    plt.imshow(img_norm, cmap='gray')
    plt.title(f"Predicted Digit: {predicted_digit} ({confidence:.2f}% Confidence)")
    plt.axis('off')
    plt.show()
    return predicted_digit

sample_img = Image.fromarray(X_test[3])
sample_img.save("new_digit.png")
predicted_digit = predict_custom_image("new_digit.png", model)
print(f"Predicted Digit: {predicted_digit}")