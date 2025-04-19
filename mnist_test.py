import numpy
import random
from pathlib import Path
from PIL import Image
from neural_network import NeuralNetwork, leaky_relu, softmax

if __name__ == "__main__":
  matrices = numpy.load("mnist_weights.npz")
  nn = NeuralNetwork(
    hidden_activation_func=leaky_relu,
    output_activation_func=softmax,
  )
  nn.load_from_file("mnist_weights.npz")
  
  folder = Path(f"./mnist/test_set")
  files = [f for f in folder.iterdir() if f.is_file()]
  random_file = random.choice(files)
  image = Image.open(random_file)
  pixels = numpy.array(image).flatten() / 255.0
  inputs = pixels.reshape((784, 1))
  
  image.show()
  
  numpy.set_printoptions(suppress=True)
  prediction = nn.predict(inputs)
  print(prediction)