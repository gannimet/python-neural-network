from PIL import Image
import numpy
import random
from neural_network import NeuralNetwork, leaky_relu, softmax, sigmoid
import utils

def load_mnist_training_data(autoencoding=False):
    training_data = []
    
    for label in range(10):
        files = utils.load_mnist_training_files(label)
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
    n_iterations = 5_000
    sample_size = 32
    structure = [784, 64, 32, 16, 16, 16, 10]
    learning_rate = 0.02
    hidden_activation_func = leaky_relu
    save_every_1k = True
    autoencoding = False
    
    print("Loading training data …")
    training_data = load_mnist_training_data(autoencoding=autoencoding)
    
    nn = NeuralNetwork(
        structure=structure,
        eta=learning_rate,
        hidden_activation_func=hidden_activation_func,
        output_activation_func=softmax,
        n_iterations=n_iterations,
        batch_size=sample_size,
        save_every_1k=save_every_1k
    )
    
    print("Training network …")
    nn.train(training_data)
    
    folder = "autoencoder_models" if autoencoding else "classification_models"
    nn.save_to_file(f"{folder}/mnist_weights_i{n_iterations}_s{sample_size}_{utils.get_layer_descriptor(nn.structure[1:-1])}.npz")
    nn.plot()