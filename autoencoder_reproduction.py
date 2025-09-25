import numpy
from neural_network import NeuralNetwork, leaky_relu
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import utils

nn = NeuralNetwork(
    hidden_activation_func=leaky_relu,
    output_activation_func=leaky_relu,
)

n_iterations = 100_000
sample_size = 32
inner_structure = [64, 32, 16, 32, 64]

nn.load_from_file(f"autoencoder_models/mnist_weights_i{n_iterations}_s{sample_size}_{utils.get_layer_descriptor(inner_structure)}.npz")
mnist_test_files = utils.load_mnist_test_files()

LATENT_VALUE_MIN = -5.0
LATENT_VALUE_MAX = 5.0
LATENT_GRID_SIZE = 4
LATENT_LAYER_INDEX = inner_structure.index(min(inner_structure)) + 1

class AutoencoderViewer:
    def __init__(self):
        # Größeres Fenster für drei Spalten - kompakter
        self.fig = plt.figure(figsize=(15, 6))
        
        # Drei Hauptbereiche definieren
        self.ax_original = plt.subplot(1, 3, 1)
        self.ax_latent_area = plt.subplot(1, 3, 2)  
        self.ax_reconstruction = plt.subplot(1, 3, 3)
        
        # Layout-Abstände anpassen - kompakter
        plt.subplots_adjust(bottom=0.12, top=0.92, left=0.05, right=0.95, wspace=0.3)
        
        # Slider-Container
        self.sliders = []
        
        # Latent-Bereich vorbereiten
        self.setup_latent_area()
        
        # Erstes Bild laden und anzeigen
        self.load_and_show_image()
        
        # Buttons für neues Bild, Reset und "Set all to 0"
        ax_button = plt.axes([0.28, 0.02, 0.15, 0.06])
        self.button = widgets.Button(ax_button, 'New random image')
        self.button.on_clicked(self.on_button_click)
        
        ax_reset = plt.axes([0.45, 0.02, 0.1, 0.06])
        self.reset_button = widgets.Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(self.on_reset_click)
        
        ax_zero = plt.axes([0.57, 0.02, 0.12, 0.06])
        self.zero_button = widgets.Button(ax_zero, 'Set all to 0')
        self.zero_button.on_clicked(self.on_zero_click)
    
    def setup_latent_area(self):
        """Erstellt die mittlere Spalte mit vertikalen Mini-Slidern"""
        self.ax_latent_area.clear()
        self.ax_latent_area.axis('off')
        
        # Container für vertikale Slider
        self.sliders = []
        
        # 4x4 Grid von vertikalen Mini-Slidern erstellen
        for i in range(nn.structure[LATENT_LAYER_INDEX]):
            row = i // LATENT_GRID_SIZE
            col = i % LATENT_GRID_SIZE
            
            # Kompakte Positionen für vertikale Slider
            slider_width = 0.015   # Sehr schmal
            slider_height = 0.08   # Relativ hoch für vertikale Orientierung
            
            # Zentriert in der mittleren Spalte
            base_x = 0.38 + (col * 0.06)  # Horizontaler Abstand zwischen Slidern
            base_y = 0.65 - (row * 0.12)  # Vertikaler Abstand zwischen Zeilen
            
            # Vertikaler Slider
            ax_slider = plt.axes([base_x, base_y, slider_width, slider_height])
            
            slider = widgets.Slider(
                ax_slider, 
                '',                 # Kein Label
                LATENT_VALUE_MIN,   # -5.0
                LATENT_VALUE_MAX,   # 5.0
                valinit=0.0,
                orientation='vertical',
                valfmt='%.1f'       # Kürzeres Format für Platzersparnis
            )
            
            # Callback für Slider-Änderung hinzufügen
            slider.on_changed(lambda val, idx=i: self.on_slider_change())
            
            self.sliders.append(slider)
    
    def load_and_show_image(self):
        """Lädt ein Bild und zeigt es in beiden Spalten"""
        # Zufälliges Bild laden
        original_pil, prediction = utils.load_random_image_and_prediction(mnist_test_files, nn)
        
        # Latent-Aktivierungen des Originalbildes abrufen (Bias bei 0 überspringen)
        exclusive_end_index = nn.structure[LATENT_LAYER_INDEX] + 1
        latent_activations = nn.activations[LATENT_LAYER_INDEX][1:exclusive_end_index].flatten()
        
        # Ursprüngliche Werte speichern für Reset-Funktion
        self.original_latent_values = latent_activations.copy()  
        
        # Slider mit Latent-Werten aktualisieren
        for i, activation_value in enumerate(latent_activations):
            # Werte auf den Slider-Bereich begrenzen
            clamped_value = numpy.clip(activation_value, LATENT_VALUE_MIN, LATENT_VALUE_MAX)
            self.sliders[i].set_val(clamped_value)
        
        # Original anzeigen (linke Spalte)
        self.ax_original.clear()
        self.ax_original.imshow(numpy.array(original_pil), cmap='gray')
        self.ax_original.set_title('Original Image')
        self.ax_original.axis('off')
        
        # Rekonstruktion anzeigen (rechte Spalte)
        self.ax_reconstruction.clear()
        reconstruction_pil = utils.create_image_from_prediction(prediction)
        self.ax_reconstruction.imshow(numpy.array(reconstruction_pil), cmap='gray')
        self.ax_reconstruction.set_title('Autoencoder Reconstruction')
        self.ax_reconstruction.axis('off')
        
        # Display aktualisieren
        self.fig.canvas.draw()
    
    def on_slider_change(self):
        """Callback für Slider-Änderungen - berechnet neue Rekonstruktion"""
        # Alle 16 Slider-Werte sammeln
        latent_copy = numpy.array([slider.val for slider in self.sliders]).reshape(-1, 1)
        
        # Neue Prediction ab Latent-Layer berechnen
        prediction = nn.predict(latent_copy, LATENT_LAYER_INDEX)
        
        # Nur das Rekonstruktionsbild aktualisieren
        self.ax_reconstruction.clear()
        reconstruction_pil = utils.create_image_from_prediction(prediction)
        self.ax_reconstruction.imshow(numpy.array(reconstruction_pil), cmap='gray')
        self.ax_reconstruction.set_title('Modified Reconstruction')
        self.ax_reconstruction.axis('off')
        
        # Effizientes Update ohne komplettes Neuzeichnen
        self.fig.canvas.draw_idle()
    
    def on_button_click(self, event):
        """Callback für 'New random image' Button"""
        self.load_and_show_image()
    
    def on_reset_click(self, event):
        """Callback für Reset-Button - setzt Slider auf ursprüngliche Werte zurück"""
        for i, original_value in enumerate(self.original_latent_values):
            clamped_value = numpy.clip(original_value, LATENT_VALUE_MIN, LATENT_VALUE_MAX)
            self.sliders[i].set_val(clamped_value)

    def on_zero_click(self, event):
        """Callback für 'Set all to 0' Button - setzt alle Slider auf 0"""
        for slider in self.sliders:
            slider.set_val(0.0)
    
    def show(self):
        """Zeigt das Fenster an"""
        plt.show()

if __name__ == "__main__":
    viewer = AutoencoderViewer()
    viewer.show()