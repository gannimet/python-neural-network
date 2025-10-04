
import numpy
from neural_network import NeuralNetwork, leaky_relu
import matplotlib.pyplot as plt
import utils
print("Importing Scikit Learn …")
from sklearn.manifold import TSNE
print("Done.")

nn = NeuralNetwork(
    hidden_activation_func=leaky_relu,
    output_activation_func=leaky_relu,
)

n_iterations = 10_000
sample_size = 32
inner_structure = [64, 32, 16, 16, 16]

nn.load_from_file(f"classification_models/mnist_weights_i{n_iterations}_s{sample_size}_{utils.get_layer_descriptor(inner_structure)}.npz")

LATENT_LAYER_INDEX = inner_structure.index(min(inner_structure)) + 1
PERPLEXITY = 10
SAMPLES_PER_DIGIT = 400

class LatentSpaceVisualizer:
    def __init__(self, nn, samples_per_digit):
        self.nn = nn
        self.samples_per_digit = samples_per_digit
        self.num_neurons = nn.structure[LATENT_LAYER_INDEX]
        
        self.latent_vectors = []
        self.labels = []
        
        self.load_samples()
        self.compute_tsne()
        self.create_plot()
    
    def load_samples(self):
        for digit in range(10):
            mnist_files = utils.load_mnist_training_files(digit)
            
            for _ in range(self.samples_per_digit):
                utils.load_random_image_and_prediction(mnist_files, self.nn)
                latent_vector = self.nn.activations[LATENT_LAYER_INDEX][1:self.num_neurons+1].flatten()
                
                self.latent_vectors.append(latent_vector)
                self.labels.append(digit)
        
        # In numpy arrays umwandeln
        self.latent_vectors = numpy.array(self.latent_vectors)
        self.labels = numpy.array(self.labels)
    
    def compute_tsne(self):
        tsne = TSNE(
            n_components=2,
            perplexity=PERPLEXITY,
            random_state=42,
            max_iter=1000
        )
        
        self.embedded = tsne.fit_transform(self.latent_vectors)
    
    def create_plot(self):
        """Erstellt die Visualisierung"""
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        
        # Farben für die Digits
        colors = plt.cm.tab10(range(10))
        
        # Jeden Digit als eigenen Scatter-Plot
        for digit in range(10):
            mask = self.labels == digit
            self.ax.scatter(
                self.embedded[mask, 0],
                self.embedded[mask, 1],
                c=[colors[digit]],
                label=str(digit),
                alpha=0.6,
                s=20
            )
        
        self.ax.set_title(f't-SNE Visualization of Latent Space ({self.samples_per_digit} samples per digit)')
        self.ax.set_xlabel('t-SNE Component 1')
        self.ax.set_ylabel('t-SNE Component 2')
        self.ax.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')
        self.ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def show(self):
        plt.show()


# Verwendung
if __name__ == "__main__":
    visualizer = LatentSpaceVisualizer(
        nn=nn,
        samples_per_digit=SAMPLES_PER_DIGIT
    )
    visualizer.show()