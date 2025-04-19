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
    outputs[label, 0] = 1
    
    for i in range(len(images)):
      pixels = numpy.array(images[i]).flatten() / 255.0
      inputs = pixels.reshape((784, 1))
      training_data.append((inputs, outputs))
      
  random.shuffle(training_data)
  return training_data
    
if __name__ == "__main__":
  print("Loading training data …")
  training_data = load_mnist_training_data()
  
  nn = NeuralNetwork(
    structure=[784, 16, 16, 16, 10],
    eta=0.01,
    hidden_activation_func=leaky_relu,
    output_activation_func=softmax,
    n_iterations=100_000
  )
  
  print("Training network …")
  nn.train(training_data)
  
  numpy.savez("mnist_weights.npz", *nn.weights)
  nn.plot()