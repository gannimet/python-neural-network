# Ein neuronales Netz in Python

Hier gibt es alle Inhalte meines VHS-Kurses „Künstliche Intelligenz selbst gemacht – Ein neuronales Netz mit Python entwickeln”. Ich nehme im Laufe der Zeit immer wieder kleine Änderungen am Kurs, dem Code und den Folien vor, korrigiere Fehler oder füge Inhalte hinzu. Es lohnt sich also, das Repository im Blick zu behalten, wenn ihr weiterhin am Thema interessiert seid.

Den Foliensatz findet ihr in der Datei [neuronale_netze.pdf](neuronale_netze.pdf).

## Fragen oder Verbesserungsvorschläge?

Kontaktiert mich gerne per Mail an [info@colimit.de](mailto:info@colimit.de). Ihr findet mich auch auf [LinkedIn](https://www.linkedin.com/in/richard-wotzlaw-8653b688/) und auf [meiner Webseite](https://colimit.de).

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
- `structure`: Die Struktur des Netzwerks als Liste von Zahlen, die jeweils die Anzahl der Neuronen pro Layer angeben. Da die Bilder 784 Pixel haben und in zehn Klassen eingeteilt werden sollen, müssen die erste und die letzte<sup>1</sup> Zahl der Liste so bleiben, mit den Neuronenzahlen der Hidden-Layer könnt ihr beliebig experimentieren.
- `learning_rate`: Die Lernrate $\eta$ des neuronalen Netzes
- `hidden_activation_func`: Die Aktivierungsfunktion, die in den Hidden-Layers verwendet werden soll
- `save_every_1k`: Wenn `True`, dann wird jeweils nach 1000 Trainingsiterationen eine Datei mit den bis dahin ermittelten Gewichten angelegt.
- `autoencoding`: Wenn `True`, werden die Trainingsdaten so angelegt, dass ein Autoencoder trainiert werden kann, der lernt, die Bilder, die er als Inputs bekommt, als seinen eigenen Output vorauszusagen.

<sup>1</sup> Im Falle, dass ihr einen Autoencoder trainieren wollt, muss die letzte Zahl der `structure`-Liste ebenfalls 784 sein

Nach beendetem Training werden die trainierten Gewichte im Ordner `classification_models` (bzw. `autoencoder_models`, falls ihr `autoencoding` auf `True` gestellt habt) als `.npz`-Datei(en) angelegt. Diese Dateien enthalten die `numpy`-Gewichtsmatrizen und können verwendet werden, um ein `NeuralNetwork`-Objekt mithilfe der Methode `load_from_file` in einen bereits trainierten Zustand zu versetzen, ohne die langwierige Berechnung wiederholen zu müssen.

Genau auf diese Weise werden die `.npz`-Dateien von den Skripten `mnist_test_prediction.py` und `mnist_draw_prediction.py` verwendet. Die Namen der Dateien starten jeweils mit `mnist_weights` und enthalten dann mit Unterstrichen getrennt noch drei weitere Informationen:

- Wie viele Iterationen gemacht wurden, um die Gewichte in der Datei zu ermitteln (Zahl hinter dem `i`)
- Wie große die Batch-Größe bei jeder Trainingsiteration war (Zahl hinter dem `s`)
- Die Anzahl der Neuronen in den Hidden-Layers (z. B. bedeutet `64x32x16`, dass es drei Hidden-Layer mit jeweils 64, 32 und 16 Neuronen gab)

### [mnist_test_prediction.py](mnist_test_prediction.py)

Wenn im Ordner `classification_models` Dateien mit trainierten Modellen liegen, könnt ihr diese benutzen, um euch vorhersagen für Test-Bilder aus dem MNIST-Datensatz anzeigen zu lassen. Dazu startet ihr einfach dieses Skript. Auf der Konsole erscheint dann eine Abfrage, welche Modell-Datei ihr laden wollt und darunter eine Liste aller gefundenen `.npz`-Dateien. Navigiert einfach mit den Pfeiltasten durch die Liste und wählt mit Enter diejenige aus, deren Gewichte ihr verwenden wollt.

Es erscheint ein Fenster, in dem ihr im linken Bereich ein zufällig geladenes Bild aus dem Testdatensatz seht und rechts davon ein Balken-Diagramm mit der Vorhersage des neuronalen Netzes – also den Softmax-Aktivierungen des letzten Layers. Über den Button „New random image” könnt ihr ein neues zufälliges Testbild laden und dessen Vorhersage anzeigen.

### [mnist_draw_prediction.py](mnist_draw_prediction.py)

Funktioniert genauso wie [mnist_test_prediction.py](mnist_test_prediction.py), nur dass hier keine Bilder aus dem MNIST-Datensatz geladen werden, sondern ihr in die schwarze Fläche links selbst eine Zahl mit dem Mauszeiger malen könnt. Die Vorhersage rechts updatet sich automatisch bei jeder Änderung.