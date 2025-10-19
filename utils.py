import random
from PIL import Image
import numpy
from pathlib import Path
import inquirer

def get_layer_descriptor(inner_structure):
    return "x".join(map(str, inner_structure))
  
def load_mnist_test_files():
    mnist_test_folder = Path("./mnist/test_set")
    return [f for f in mnist_test_folder.iterdir() if f.suffix == '.jpg']

def load_mnist_training_files(label):
    mnist_training_folder = Path(f"./mnist/training_set/{label}")
    return [f for f in mnist_training_folder.iterdir() if f.suffix == '.jpg']
  
def load_random_image_and_prediction(image_files, network):
    random_file = random.choice(image_files)
    image = Image.open(random_file)
    pixels = numpy.array(image).flatten() / 255.0
    inputs = pixels.reshape((784, 1))
    prediction = network.predict(inputs)
    return image, prediction

def create_image_from_prediction(prediction):
    prediction_reshaped = prediction.reshape((28, 28))
    prediction_clipped_and_scaled = numpy.clip(prediction_reshaped, 0, 1) * 255.0
    return Image.fromarray(prediction_clipped_and_scaled.astype(numpy.uint8))

def select_model_file(classification=True):
    folder = Path(f"classification_models/") if classification else Path(f"autoencoder_models/")
    filenames = [f.name for f in folder.iterdir() if f.is_file() and f.suffix == ".npz"]
    filenames.sort()

    questions = [
        inquirer.List(
            'filename',
            message="Welches Modell soll geladen werden?",
            choices=filenames,
            carousel=True
        )
    ]
    answer = inquirer.prompt(questions)
    filename = folder.joinpath(answer["filename"])
    
    return filename