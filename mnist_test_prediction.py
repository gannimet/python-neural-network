import numpy
from neural_network import NeuralNetwork, leaky_relu, softmax
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import utils

nn = NeuralNetwork(
    hidden_activation_func=leaky_relu,
    output_activation_func=softmax,
)

n_iterations = 10_000
sample_size = 50
inner_structure = [128, 64, 32, 32, 32]

nn.load_from_file(f"classification_models/mnist_weights_i{n_iterations}_s{sample_size}_{utils.get_layer_descriptor(inner_structure)}.npz")
mnist_test_files = utils.load_mnist_test_files()

fig = plt.figure(1, figsize=(8, 4))
ax_img = fig.add_axes([0.05, 0.2, 0.4, 0.7])
ax_bar = fig.add_axes([0.5, 0.2, 0.45, 0.7])
ax_button = fig.add_axes([0.15, 0.05, 0.2, 0.1])

def update_display():
    ax_img.clear()
    ax_bar.clear()
    image, prediction = utils.load_random_image_and_prediction(mnist_test_files, nn)
    
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
    new_image_button = Button(ax_button, "New random image")
    new_image_button.on_clicked(on_button_click)
    
    numpy.set_printoptions(suppress=True)
    
    # Anzeigen
    plt.show()
    
def on_button_click(event):
        update_display()

if __name__ == "__main__":
    update_display()
    