from PIL import Image
import numpy
from pathlib import Path
import random
from neural_network import NeuralNetwork, leaky_relu, softmax, sigmoid

def load_mnist_training_data(autoencoding=False):
    training_data = []
    
    for label in range(10):
        folder = Path(f"./mnist/training_set/{label}")
        files = [f for f in folder.iterdir() if f.suffix == '.jpg']
        outputs = numpy.zeros((10, 1))
        outputs[label, 0] = 1.0
        
        for file in files:
            image = Image.open(file)
            inputs = numpy.array(image).reshape((784, 1)) / 255
            
            if autoencoding:
                training_data.append((inputs, inputs))
            else:
                training_data.append((inputs, outputs))
            
    random.shuffle(training_data)
    return training_data
        
if __name__ == "__main__":
    print("Loading training data …")
    training_data = load_mnist_training_data()
    
    n_iterations = 1_000_000
    sample_size = 1_000
    
    nn = NeuralNetwork(
        structure=[784, 16, 16, 16, 10],
        eta=0.02,
        hidden_activation_func=leaky_relu,
        output_activation_func=softmax,
        n_iterations=n_iterations,
        batch_size=sample_size
    )
    
    print("Training network …")
    nn.train(training_data)
    
    nn.save_to_file(f"classification_models/mnist_weights_i{n_iterations}_s{sample_size}.npz")
    nn.plot()