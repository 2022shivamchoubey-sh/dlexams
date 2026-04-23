# Experiments Code

codes = {
    "exp1": """import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])
model = keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    layers.Dense(4, activation='tanh'),
    layers.Dense(1, activation='sigmoid')
    ])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=10, verbose=1)

predictions = model.predict(X)

print("Predictions:")
print(predictions)


print("\nRounded Predictions:")
print(np.round(predictions))""",
    "exp2": """import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(test_loss)
print(test_accuracy)

plt.plot(model_history.history['loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()""",
    "exp3": """import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

x_train = x_train/255.0
x_test = x_test/255.0

model = keras.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

eval = model.evaluate(x_test, y_test)
print(eval)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2)

axes[0].plot(model_history.history['accuracy'])
axes[0].set_title("Model accuracy")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Accuracy")

axes[1].plot(model_history.history['loss'])
axes[1].set_title("Model loss")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("loss")

model2 = keras.Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation='tanh'),
    layers.Dense(64, activation='tanh'),
    layers.Dense(10, activation='softmax')
])
model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model2_history = model2.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2) # Added missing ')' and changed batch_size to 32
eval2 = model2.evaluate(x_test, y_test)
print(eval2)

# The variable model_history2 was not defined in the original code, but model2_history was.
# Assuming the intent was to plot model2_history, I will use that.
fig, axes = plt.subplots(1, 2)

axes[0].plot(model2_history.history['accuracy'])
axes[0].set_title("Model accuracy")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Accuracy")

axes[1].plot(model2_history.history['loss'])
axes[1].set_title("Model loss")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("loss")""",
    "exp4": """import tensorflow as tf

# 1. Load Data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Build Model (Using CNN architecture)
model = tf.keras.Sequential([
tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(10, activation='softmax')
])

# 3. Compile & Fit (With Callbacks)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callbacks = [
tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**-(epoch / 10))
]
history=model.fit(x_train, y_train, epochs=20, validation_split=0.2, callbacks=callbacks)
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
""",
    "exp5": """import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Define the plot function
def plot(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.show()

# 1. Load Data
dataset = tfds.load('svhn_cropped', split='train', as_supervised=True)
train_ds = dataset.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(64)

# 2. Build Model (Adding requested layers)
model = tf.keras.Sequential([
tf.keras.layers.Input(shape=(32, 32, 3)), # Explicit Input layer
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(256, activation='relu'),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(10, activation='softmax')
])

# 3. Compile & Fit
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=5)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_ds,epochs=5)
plot(history)""",
    "exp6": """import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data & Add Noise
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train_noisy = np.clip(x_train + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape), 0., 1.)

# 2. Build Model (Encoder -> Decoder)
model = tf.keras.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(784, activation='sigmoid'),
tf.keras.layers.Reshape((28, 28))
])

# 3. Compile & Fit
model.compile(optimizer='adam', loss='mse')
history = model.fit(x_train_noisy, x_train, epochs=5, validation_split=0.2)

# 4. Plot
plt.plot(history.history['loss'], label='Reconstruction Loss')
plt.legend()
plt.show()""",
    "exp7": """import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Define your plotting function
def plot(history):
    plt.plot(history.history['loss'], label='loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='val_loss')
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='acc')
        if 'val_accuracy' in history.history: # Added safety check for val_accuracy
            plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()

# 1. Load Data (Predicting if the person is smiling)
train_ds = tfds.load('celeb_a', split='train').map(
    lambda d: (tf.cast(d['image'], tf.float32)/255.0, d['attributes']['Smiling'])
).batch(64)

# 2. Build Model (CNN)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(218, 178, 3)), # Recommended syntax for Input
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 3. Compile & Fit
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Save the training output to the 'history' variable
history = model.fit(train_ds, epochs=3)

# 4. Call your plot function
plot(history)""",
    "exp8": """import tensorflow as tf
# 1. Load Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# 2. Build Model (Optimized Architecture)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])
# 3. Compile & Fit
import matplotlib.pyplot as plt
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()""",
}

def show(exp_name=None, run=False):
    """
    Shows and optionally injects/runs deep learning experiments.
    In Colab, this injects the code into the next cell automatically!
    """
    if exp_name is None:
        print("Available experiments:")
        for key in codes.keys():
            print(f"- {key}")
        print("\nUsage: show('exp1') to print and inject into a Colab cell, or show('exp1', run=True) to execute immediately.")
        return

    if exp_name not in codes:
        print(f"❌ Experiment '{exp_name}' not found. Available: {', '.join(codes.keys())}")
        return

    code = codes[exp_name]
    
    if run:
        print(f"🚀 Running {exp_name} immediately...")
        exec(code, globals())
        return

    print(f"--- Code for {exp_name} ---")
    print(code)
    
    # Try to inject code into the next Colab cell
    try:
        from IPython.core.getipython import get_ipython
        shell = get_ipython()
        if shell is not None:
            shell.set_next_input(code, replace=False)
            print("\n✅ Code injected into the next cell. Press Shift+Enter to run it!")
        else:
            print("\n(Note: Cell injection only works in Jupyter/Colab notebooks.)")
    except ImportError:
        pass
