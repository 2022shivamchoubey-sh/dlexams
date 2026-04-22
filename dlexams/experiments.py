# Experiments Code

# This dictionary contains all experiments and their implementations in TensorFlow/Keras.

codes = {
    'exp1': 'MLP XOR code here',
    'exp2': 'MNIST training code here',
    'exp3': 'Fashion MNIST comparison code here',
    'exp4': 'CIFAR-10 with learning rate schedule code here',
    'exp5': 'SVHN with batch normalization and dropout code here',
    'exp6': 'Denoising autoencoder code here',
    'exp7': 'CelebA CNN code here',
    'exp8': 'Optimized Fashion MNIST MLP code here'
}

# Function to show the experiments

def show():
    for key, value in codes.items():
        print(f'Experiment {key}: {value}')