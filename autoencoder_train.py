from neural_network import NeuralNetwork, leaky_relu, sigmoid
from mnist_train import load_mnist_training_data
import utils

if __name__ == "__main__":
    print("Loading training data …")
    training_data = load_mnist_training_data(autoencoding=True)
    
    n_iterations = 100
    sample_size = 32
    
    nn = NeuralNetwork(
        structure=[784, 64, 32, 16, 32, 64, 784],
        eta=0.003,
        hidden_activation_func=leaky_relu,
        output_activation_func=leaky_relu,
        n_iterations=n_iterations,
        batch_size=sample_size,
        save_every_1k=False
    )
    
    print("Training network …")
    nn.train(training_data)
    
    nn.save_to_file(f"autoencoder_models/mnist_weights_i{n_iterations}_s{sample_size}_{utils.get_layer_descriptor(nn.structure[1:-1])}.npz")
    nn.plot()