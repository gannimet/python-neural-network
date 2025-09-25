import numpy
from neural_network import NeuralNetwork, leaky_relu
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import utils
import random

nn = NeuralNetwork(
    hidden_activation_func=leaky_relu,
    output_activation_func=leaky_relu,
)

n_iterations = 100_000
sample_size = 32
inner_structure = [64, 32, 16, 32, 64]

nn.load_from_file(f"autoencoder_models/mnist_weights_i{n_iterations}_s{sample_size}_{utils.get_layer_descriptor(inner_structure)}.npz")

LATENT_LAYER_INDEX = inner_structure.index(min(inner_structure)) + 1

class DigitGridViewer:
    def __init__(self):
        # Fenster erstellen - kompakter
        self.fig = plt.figure(figsize=(10, 6))
        
        # Layout-Abstände optimieren - kompakter
        plt.subplots_adjust(bottom=0.05, top=0.92, left=0.05, right=0.95, wspace=0.15)
        
        # Buttons für Ziffern 0-9 erstellen
        self.digit_buttons = []
        self.setup_digit_buttons()
        
        # Zwei-Spalten-Layout unter den Buttons
        self.setup_columns()
    
    def setup_digit_buttons(self):
        """Erstellt eine Reihe von Buttons für die Ziffern 0-9"""
        button_width = 0.06   # Kleinere Buttons
        button_height = 0.05
        total_width = 10 * button_width + 9 * 0.01  # 10 Buttons + 9 Abstände
        start_x = (1.0 - total_width) / 2  # Zentriert
        y_position = 0.85  # Weiter oben
        
        for digit in range(10):
            # Position für jeden Button berechnen - zentriert
            x_position = start_x + (digit * (button_width + 0.01))
            
            # Button erstellen
            ax_button = plt.axes([x_position, y_position, button_width, button_height])
            button = widgets.Button(ax_button, str(digit))
            button.on_clicked(lambda event, d=digit: self.on_digit_click(d))
            
            self.digit_buttons.append(button)
    
    def setup_columns(self):
        """Erstellt das zwei-Spalten-Layout unter den Buttons"""
        # Spalte A: 4x4 Grid für kleine Bilder
        self.column_a_axes = []
        grid_size = 4
        
        # Startposition und Größe für das 4x4 Grid - kompakter
        grid_start_x = 0.08
        grid_start_y = 0.08
        grid_width = 0.35   # Schmaler
        grid_height = 0.65  # Höher
        
        # Einzelne Felder im 4x4 Grid erstellen
        for row in range(grid_size):
            for col in range(grid_size):
                # Position jedes Grid-Felds berechnen
                field_width = grid_width / grid_size * 0.85  # Weniger Abstand
                field_height = grid_height / grid_size * 0.85
                
                x_pos = grid_start_x + (col * grid_width / grid_size) + (grid_width / grid_size * 0.075)
                y_pos = grid_start_y + ((grid_size - 1 - row) * grid_height / grid_size) + (grid_height / grid_size * 0.075)
                
                # Subplot für jedes Grid-Feld
                ax = plt.axes([x_pos, y_pos, field_width, field_height])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal')
                
                # Rahmen um leere Felder für bessere Sichtbarkeit
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='gray', linewidth=1))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                
                self.column_a_axes.append(ax)
        
        # Spalte B: Großes Bild - näher an Spalte A
        self.column_b_ax = plt.axes([0.52, grid_start_y, grid_width, grid_height])
        self.column_b_ax.set_xticks([])
        self.column_b_ax.set_yticks([])
        self.column_b_ax.set_aspect('equal')
        
        # Rahmen um das große Feld
        self.column_b_ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor='gray', linewidth=2))
        self.column_b_ax.set_xlim(0, 1)
        self.column_b_ax.set_ylim(0, 1)
        
        # Labels für die Spalten - näher an den Inhalten
        self.fig.text(0.255, 0.75, 'Random sample images', ha='center', fontsize=11, weight='bold')
        self.fig.text(0.695, 0.75, 'Reproduction of average latent vectors', ha='center', fontsize=11, weight='bold')
    
    def on_digit_click(self, digit):
        mnist_digit_files = utils.load_mnist_training_files(digit)
        num_neurons = nn.structure[LATENT_LAYER_INDEX]
        average_latent = numpy.zeros((num_neurons, 1))
        
        for i in range(16):
            image, _ = utils.load_random_image_and_prediction(mnist_digit_files, nn)
            self.column_a_axes[i].clear()
            self.column_a_axes[i].imshow(image, cmap='gray')
            self.column_a_axes[i].axis('off')
            average_latent += nn.activations[LATENT_LAYER_INDEX][1:num_neurons+1]

        average_latent /= 16
        prediction_of_average = nn.predict(average_latent, LATENT_LAYER_INDEX)
        average_image = utils.create_image_from_prediction(prediction_of_average)

        self.column_b_ax.clear()
        self.column_b_ax.imshow(average_image, cmap='gray')
        self.column_b_ax.axis('off')

        self.fig.canvas.draw_idle()
    
    def show(self):
        """Zeigt das Fenster an"""
        plt.show()

# Verwendung:
if __name__ == "__main__":
    viewer = DigitGridViewer()
    viewer.show()