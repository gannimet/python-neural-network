import numpy
import random
from PIL import Image
from neural_network import NeuralNetwork, leaky_relu
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

nn = NeuralNetwork(
    hidden_activation_func=leaky_relu,
    output_activation_func=leaky_relu,
)

n_iterations = 100_000
sample_size = 32

nn.load_from_file(f"autoencoder_models/mnist_weights_i{n_iterations}_s{sample_size}_256x128x64x128x256.npz")
folder = Path(f"./mnist/test_set")
files = [f for f in folder.iterdir() if f.is_file()]

def load_random_image_and_prediction():
    random_file = random.choice(files)
    image = Image.open(random_file)
    pixels = numpy.array(image).flatten() / 255.0
    inputs = pixels.reshape((784, 1))
    prediction = nn.predict(inputs)
    return image, prediction

def create_image_from_prediction(prediction):
    prediction_reshaped = prediction.reshape((28, 28))
    prediction_clipped_and_scaled = numpy.clip(prediction_reshaped, 0, 1) * 255.0
    return Image.fromarray(prediction_clipped_and_scaled.astype(numpy.uint8), mode='L')

class AutoencoderViewer:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        plt.subplots_adjust(bottom=0.2)
        
        self.update_images()
        
        ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.button = widgets.Button(ax_button, 'New random image')
        self.button.on_clicked(self.on_button_click)
        
        self.ax1.set_title('Original Image')
        self.ax2.set_title('Autoencoder Reconstruction')
        self.ax1.axis('off')
        self.ax2.axis('off')
    
    def update_images(self):
        original_pil, prediction = load_random_image_and_prediction()
        reconstruction_pil = create_image_from_prediction(prediction)
        
        original_array = numpy.array(original_pil)
        reconstruction_array = numpy.array(reconstruction_pil)
        
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.imshow(original_array, cmap='gray')
        self.ax1.set_title('Original Image')
        self.ax1.axis('off')
        
        self.ax2.imshow(reconstruction_array, cmap='gray')
        self.ax2.set_title('Autoencoder Reconstruction')
        self.ax2.axis('off')
        
        self.fig.canvas.draw()
    
    def on_button_click(self, event):
        self.update_images()
    
    def show(self):
        plt.show()

viewer = AutoencoderViewer()
viewer.show()