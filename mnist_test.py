import numpy
import random
from pathlib import Path
from PIL import Image
from neural_network import NeuralNetwork, leaky_relu, softmax
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

nn = NeuralNetwork(
  hidden_activation_func=leaky_relu,
  output_activation_func=softmax,
)
n_iterations = 1_000_000
sample_size = 2000
nn.load_from_file(f"trained_models/mnist_weights_i{n_iterations}_s{sample_size}.npz")
folder = Path(f"./mnist/test_set")
files = [f for f in folder.iterdir() if f.is_file()]

fig = plt.figure(1, figsize=(8, 4))
ax_img = fig.add_axes([0.05, 0.2, 0.4, 0.7])
ax_bar = fig.add_axes([0.5, 0.2, 0.45, 0.7])
ax_button = fig.add_axes([0.15, 0.05, 0.2, 0.1])

def load_random_image_and_prediction():
  random_file = random.choice(files)
  image = Image.open(random_file)
  pixels = numpy.array(image).flatten() / 255.0
  inputs = pixels.reshape((784, 1))
  prediction = nn.predict(inputs)
  return image, prediction

def update_display():
  ax_img.clear()
  ax_bar.clear()
  image, prediction = load_random_image_and_prediction()
  
  # Bild
  ax_img.imshow(image, cmap='gray')
  ax_img.axis('off')
  ax_img.set_title("Image")

  # Balkendiagramm
  max_index = numpy.argmax(prediction)
  colors = ['paleturquoise' if i != max_index else 'teal' for i in range(len(prediction))]
  ax_bar.bar(range(len(prediction)), prediction.flatten(), color=colors)
  ax_bar.set_ylim(0, 1)
  ax_bar.set_xticks(range(len(prediction)))
  ax_bar.set_xlabel("Predicted class")
  ax_bar.set_ylabel("Confidence")
  ax_bar.set_title("Prediction")
  
  # Button
  button = Button(ax_button, "New random image")
  button.on_clicked(on_button_click)
  
  numpy.set_printoptions(suppress=True)
  numpy.savez(f"trained_activations/activations_{max_index}.npz", *nn.activations, *nn.weights)
  
  # Anzeigen
  plt.show()
  
def on_button_click(event):
    update_display()

if __name__ == "__main__":
  update_display()
  