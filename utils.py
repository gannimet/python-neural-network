import random
from PIL import Image
import numpy
from neural_network import NeuralNetwork
from pathlib import Path

def get_layer_descriptor(inner_structure):
    return "x".join(map(str, inner_structure))
  
def load_mnist_test_files():
    mnist_test_folder = Path(f"./mnist/test_set")
    return [f for f in mnist_test_folder.iterdir() if f.suffix == '.jpg']

def load_mnist_training_files(label):
    mnist_training_folder = Path(f"./mnist/training_set/{label}")
    return [f for f in mnist_training_folder.iterdir() if f.suffix == '.jpg']
  
def load_random_image_and_prediction(mnist_test_files: list[Path], network: NeuralNetwork):
    random_file = random.choice(mnist_test_files)
    image = Image.open(random_file)
    pixels = numpy.array(image).flatten() / 255.0
    inputs = pixels.reshape((784, 1))
    prediction = network.predict(inputs)
    return image, prediction