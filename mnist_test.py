import numpy
import random
from pathlib import Path
from PIL import Image
from neural_network import NeuralNetwork, leaky_relu, softmax
import matplotlib.pyplot as plt

if __name__ == "__main__":
  nn = NeuralNetwork(
    hidden_activation_func=leaky_relu,
    output_activation_func=softmax,
  )
  nn.load_from_file("trained_models/mnist_weights_i100000_s2000.npz")
  
  folder = Path(f"./mnist/test_set")
  files = [f for f in folder.iterdir() if f.is_file()]
  random_file = random.choice(files)
  image = Image.open(random_file)
  pixels = numpy.array(image).flatten() / 255.0
  inputs = pixels.reshape((784, 1))
  prediction = nn.predict(inputs)

  fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(8, 4))

  # Bild anzeigen
  ax_img.imshow(image, cmap='gray')
  ax_img.axis('off')
  ax_img.set_title("Bild")

  # Balkendiagramm anzeigen
  ax_bar.bar(range(len(prediction)), prediction.flatten(), color='skyblue')
  ax_bar.set_ylim(0, 1)
  ax_bar.set_xticks(range(len(prediction)))
  ax_bar.set_xlabel("Klasse")
  ax_bar.set_ylabel("Wahrscheinlichkeit")
  ax_bar.set_title("Vorhersage")
  
  numpy.set_printoptions(suppress=True)
  print(prediction)

  # Anzeigen
  plt.tight_layout()
  plt.show()