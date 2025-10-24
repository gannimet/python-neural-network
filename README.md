[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/richard-wotzlaw-8653b688/)

[![Email Badge](https://img.shields.io/badge/Gmail-Contact_Me-green?style=flat-square&logo=gmail&logoColor=FFFFFF&labelColor=3A3B3C&color=62F1CD)](mailto:r.wotzlaw@gmail.com)

# Ein neuronales Netz in Python

Hier gibt es alle Inhalte meines VHS-Kurses „Künstliche Intelligenz selbst gemacht – Ein neuronales Netz mit Python entwickeln”. Ich nehme im Laufe der Zeit immer wieder kleine Änderungen am Kurs, dem Code und den Folien vor, korrigiere Fehler oder füge Inhalte hinzu. Es lohnt sich also, das Repository im Blick zu behalten, wenn ihr weiterhin am Thema interessiert seid.

Den Foliensatz findet ihr in der Datei [neuronale_netze.pdf](neuronale_netze.pdf).

## Benötigte Libraries installieren

Um den Code zum Laufen zu bringen, müsst ihr einmalig zu Beginn die Abhängigkeiten installieren. Das geht am einfachsten mit folgendem Befehl auf der Konsole. Ihr müsst dazu im Root-Verzeichnis des Projekts sein. Dort seid ihr z. B. automatisch, wenn ihr das Projekt in VSCode öffnet und dort einen neuen Terminal mit `Terminal → New Terminal` öffnet.

```console
pip install -r requirements.txt
```

## Die einzelnen Skripte im Detail

Nicht auf alle Python-Dateien gehe ich im Rahmen des Kurses ganz genau ein. Die, die für euch relevant und/oder interessant sind oder sein könnten, erkläre ich hier etwas genauer und erläutere jeweils den Zweck und wie ihr es verwenden könnt.

### [perceptron.py](perceptron.py)

Der Code für das Perceptron mit zwei Inputs, den wir gemeinsam im Kurs entwickeln. Die Datei enthält sowohl die Klasse als auch den Code, der sie verwendet, um ein Perceptron auf einen Datensatz aus insgesamt 12 Punkten zu trainieren. Wenn ihr die Datei ausführt, öffnet sich ein Fenster, dass nach erfolgreichem Training die ermittelte Trennlinie durch die zwei Punktwolken anzeigt.

### [perceptron_animated.py](perceptron_animated.py)

Macht das Gleiche wie [perceptron.py](perceptron.py), animiert aber die einzelnen Iterationsschritte und zeigt, wie sich die Linie nach und nach der Lösung annähert.

### [neural_network.py](neural_network.py)

Diese Datei ist das Herzstück und enthält die Klasse `NeuralNetwork`, die das neuronale Netz modelliert. Sie ist alleine nicht ausführbar (bzw. tut allein nichts sinnvolles), wird aber von den anderen Datein importiert. Die wichtigsten Methoden der Klasse `NeuralNetwork` sind:

- `predict(X)`: Nimmt einen Inputvektor `X` entgegen und rechnet dafür eine Prediction aus („Feedforward”-Schritt)
- `train(training_data)`: Trainiert das Netz auf den Datensatz in `training_data` mittels des „Backpropagation”-Algorithmus

### [mnist_train.py](mnist_train.py)

Wenn ihr diese Datei ausführt, lädt sie die [MNIST-Trainingsbilder](https://en.wikipedia.org/wiki/MNIST_database) aus dem Ordner `mnist` und trainiert ein `NeuralNetwork`-Objekt darauf, diesen Datensatz klassifizieren zu können. Die Parameter des Trainings könnt ihr einstellen, indem ihr in der Datei die Werte der folgenden Variablen anpasst:

- `n_iterations`: Die Anzahl der Trainingsiterationen („Epochen”), die durchgeführt werden sollen
- `sample_size`: Wie viele Trainingsbeispiele bei jeder Iteration verwendet werden sollen
- `structure`: Die Struktur des Netzwerks als Liste von Zahlen, die jeweils die Anzahl der Neuronen pro Layer angeben. Da die Bilder 784 Pixel haben und in zehn Klassen eingeteilt werden sollen, müssen die erste und die letzte Zahl der Liste so bleiben, mit den Neuronenzahlen der Hidden-Layer könnt ihr beliebig experimentieren.
- `learning_rate`: Die Lernrate $\eta$ des neuronalen Netzes
- `hidden_activation_func`: Die Aktivierungsfunktion, die in den Hidden-Layers verwendet werden soll
- `save_every_1k`: Wenn die Variable auf `True` steht, wird jeweils nach 1000 Trainingsiterationen eine Datei mit den bis dahin ermittelten Gewichten angelegt.

Nach beendetem Training werden die trainierten Gewichte im Ordner `classification_models` als `.npz`-Datei(en) angelegt. Diese Dateien enthalten die `numpy`-Gewichtsmatrizen und können verwendet werden, um ein `NeuralNetwork`-Objekt mithilfe der Methode `load_from_file` in einen bereits trainierten Zustand zu versetzen, ohne die langwierige Berechnung wiederholen zu müssen.

Genau auf diese Weise werden die `.npz`-Dateien von den Skripten `mnist_test_prediction.py` und `mnist_draw_prediction.py` verwendet.