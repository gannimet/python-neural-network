from PIL import Image
import numpy
from pathlib import Path
import random
from neural_network import NeuralNetwork, leaky_relu, softmax

def load_mnist_training_data():
  training_data = []
  
  for label in range(10):
    folder = Path(f"./mnist/training_set/{label}")
    images = [Image.open(f) for f in folder.iterdir() if f.is_file()]
    outputs = numpy.zeros((10, 1))
    outputs[label, 0] = 1.0
    
    for i in range(len(images)):
      pixels = numpy.array(images[i]).flatten() / 255.0
      inputs = pixels.reshape((784, 1))
      training_data.append((inputs, outputs))
      
  random.shuffle(training_data)
  return training_data
    
if __name__ == "__main__":
  print("Loading training data …")
  training_data = load_mnist_training_data()
  
  n_iterations = 1_000_001
  sample_size = 1_000
  
  nn = NeuralNetwork(
    structure=[784, 16, 16, 16, 10],
    eta=0.01,
    hidden_activation_func=leaky_relu,
    output_activation_func=softmax,
    n_iterations=n_iterations,
    batch_size=sample_size
  )
  # nn = NeuralNetwork(
  #   eta=0.02,
  #   hidden_activation_func=leaky_relu,
  #   output_activation_func=softmax,
  #   n_iterations=n_iterations,
  #   batch_size=sample_size
  # )
  nn.load_from_file("trained_models/mnist_weights_i2300000_s1000.npz")
  
  print("Training network …")
  nn.train(training_data)
  
  #nn.save_to_file(f"trained_models/mnist_weights_i{n_iterations}_s{sample_size}.npz")
  nn.plot()