import numpy
from neural_network import NeuralNetwork, leaky_relu, softmax
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import utils

nn = NeuralNetwork(
    hidden_activation_func=leaky_relu,
    output_activation_func=softmax,
)

n_iterations = 5_000
sample_size = 32
inner_structure = [64, 32, 16, 16, 16]

nn.load_from_file(f"classification_models/mnist_weights_i{n_iterations}_s{sample_size}_{utils.get_layer_descriptor(inner_structure)}.npz")

# Initialisierung
fig = plt.figure(1, figsize=(8, 4))
ax_img = fig.add_axes([0.05, 0.2, 0.4, 0.7])
ax_bar = fig.add_axes([0.5, 0.2, 0.45, 0.7])
ax_button = fig.add_axes([0.15, 0.05, 0.2, 0.1])

# 28x28 schwarzes Bild (wie MNIST)
canvas = numpy.zeros((28, 28))
img_display = None
is_drawing = False

def update_prediction():
    """Aktualisiert nur das Balkendiagramm basierend auf dem aktuellen Canvas"""
    # Canvas in Input-Vektor umwandeln (normalisiert auf [0, 1])
    input_vector = canvas.flatten().reshape(-1, 1) / 255.0
    
    # Prediction berechnen
    prediction = nn.predict(input_vector, 0)
    
    # Balkendiagramm aktualisieren
    ax_bar.clear()
    max_index = numpy.argmax(prediction)
    colors = ['paleturquoise' if i != max_index else 'teal' for i in range(len(prediction))]
    ax_bar.bar(range(len(prediction)), prediction.flatten(), color=colors)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xticks(range(len(prediction)))
    ax_bar.set_xlabel("Predicted class")
    ax_bar.set_ylabel("Confidence")
    ax_bar.set_title("Prediction")
    
    # Canvas aktualisieren
    img_display.set_data(canvas)
    fig.canvas.draw_idle()

def on_mouse_press(event):
    """Startet das Zeichnen"""
    global is_drawing
    if event.inaxes == ax_img:
        is_drawing = True
        draw_on_canvas(event)

def on_mouse_release(event):
    """Stoppt das Zeichnen"""
    global is_drawing
    is_drawing = False

def on_mouse_move(event):
    """Zeichnet während der Mausbewegung"""
    if is_drawing and event.inaxes == ax_img:
        draw_on_canvas(event)

def draw_on_canvas(event):
    """Zeichnet auf dem Canvas an der Mausposition"""
    global canvas
    
    x, y = int(event.xdata), int(event.ydata)
    
    if 0 <= x < 28 and 0 <= y < 28:
        brush_size = 1
        for dx in range(0, brush_size + 1):
            for dy in range(0, brush_size + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 28 and 0 <= ny < 28:
                    canvas[ny, nx] = min(255, canvas[ny, nx] + 100)
        
        update_prediction()

def clear_canvas(event):
    """Setzt das Canvas auf schwarz zurück"""
    global canvas
    canvas = numpy.zeros((28, 28))
    update_prediction()

def setup_display():
    """Initialisiert die Anzeige"""
    global img_display
    
    # Bild
    ax_img.clear()
    img_display = ax_img.imshow(canvas, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    ax_img.axis('off')
    ax_img.set_title("Draw a digit")
    ax_img.set_xlim(-0.5, 27.5)
    ax_img.set_ylim(27.5, -0.5)  # Y-Achse invertieren für korrekte Ausrichtung
    
    # Button
    clear_button = Button(ax_button, "Clear")
    clear_button.on_clicked(clear_canvas)
    
    # Mouse-Events verbinden
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('button_release_event', on_mouse_release)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    
    # Initiale Prediction (alles schwarz)
    update_prediction()
    
    plt.show()

if __name__ == "__main__":
    setup_display()