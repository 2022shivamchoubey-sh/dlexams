# Deep Learning Experiments

## Experiment 1: MLP XOR
```python
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# XOR data
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])

# Build MLP Model
model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(data, labels, epochs=5000, verbose=0)

# Show predictions
print("XOR Predictions:", model.predict(data))
```

## Experiment 2: MNIST Training
```python
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build MLP Model
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)
```

## Experiment 3: Fashion MNIST Comparison
```python
from keras.datasets import fashion_mnist
# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize data
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build and train summary function for MLP and CNN
```

## Experiment 4: CIFAR-10 Learning Rate Schedule
```python
from keras.datasets import cifar10
# Load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Learning rate scheduler implementation needed
```

## Experiment 5: Batch Normalization and Dropout on SVHN
```python
# SVHN dataset code, including Batch Normalization and Dropout layers
```

## Experiment 6: Denoising Autoencoder on MNIST
```python
# Denoising Autoencoder implementation goes here
```

## Experiment 7: CNN on CelebA
```python
# CNN implementation for CelebA dataset code
```

## Experiment 8: Optimized MLP for Fashion MNIST
```python
# Optimized MLP model code for Fashion MNIST
```
