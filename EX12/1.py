import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.layers import Dense
from keras.models import Sequential

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Build 3-layer ANN (1 input specification + 1 hidden + 1 output)
model_3layer = Sequential(
    [Dense(8, activation="relu", input_shape=(4,)), Dense(3, activation="softmax")]
)

# 5. Compile model
model_3layer.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 6. Train model
model_3layer.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)

# 7. Evaluate
loss_a, acc_a = model_3layer.evaluate(X_test, y_test, verbose=0)
print(f"\nPart A - Test Loss: {loss_a:.4f} | Test Accuracy: {acc_a:.4f}")

# Predict sample
pred = model_3layer.predict(X_test[:1])
print(f"Actual: {y_test[0]}, Predicted: {pred.argmax()}")

# Build multi-layer ANN with 3 hidden layers
model_deep = Sequential([
    Dense(16, activation="relu", input_shape=(4,)),  # Hidden Layer 1
    Dense(16, activation="relu"),  # Hidden Layer 2
    Dense(8, activation="relu"),  # Hidden Layer 3
    Dense(3, activation="softmax"),  # Output Layer
])

# Compile model
model_deep.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Train model
model_deep.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)

# Evaluate deep model
loss_b, acc_b = model_deep.evaluate(X_test, y_test, verbose=0)
print(f"\nPart B - Test Loss: {loss_b:.4f} | Test Accuracy: {acc_b:.4f}")