import numpy
import matplotlib.pyplot as plt

def relu(x, derivative=False):
  if derivative:
    return numpy.heaviside(x, 1) # 1 wenn x>=0 ist, 0 wenn x < 0
  return numpy.maximum(x, 0)

def leaky_relu(x, derivative=False):
  if derivative:
    return numpy.where(x > 0, 1, 0.1)
  return numpy.where(x > 0, x, 0.1 * x)

def identity(x, derivative=False):
  if derivative:
    return 1
  return x

def sigmoid(x, derivative=False):
  if derivative:
    return sigmoid(x) * (1 - sigmoid(x))
  return 1 / (1 + numpy.exp(-x))

class NeuralNetwork():
  def __init__(
      self,
      structure, # Array mit Anzahlen der Neuronen pro Layer
      eta=0.01, # Lernrate
      n_iterations=1000, # Anzahl an Iterationen, die im Lernschritt durchgeführt werden sollen
      output_activation_func=identity, # Aktivierungsfunktion im Output-Layer
      hidden_activation_func=relu, # Aktivierungsfunktion in den Hidden Layers
  ):
    self.structure = structure
    self.eta = eta
    self.n_iterations = n_iterations
    self.output_activation_func = output_activation_func
    self.hidden_activation_func = hidden_activation_func
    self.error_progression = []
    self.__init_layers()
    self.__init_weights()

  def __init_layers(self):
    """
    Initialisiert alle Layer des Netzes so, wie sie durch
    den Parameter structure vorgegeben sind. Anfangs werden
    alle Aktivierungen mit 0 belegt
    """
    self.activations = []
    self.weighted_sums = []
    
    for l in range(len(self.structure)):
      is_output_layer = l == len(self.structure) - 1
      # Berechne Anzahl der Neuronen, addiere 1 für das Bias-Neuron in
      # Input- und Hidden Layer
      n_neurons = self.structure[l] if is_output_layer else self.structure[l] + 1
      # Layer für Neuronenaktivierungen ist Vektor
      # mit 1 Spalte, die mit 0en befüllt wird
      layer_weighted_sums = numpy.zeros((n_neurons, 1))
      layer_activations = numpy.zeros((n_neurons, 1))

      # Füge den Layer zum Netz hinzu
      self.weighted_sums.append(layer_weighted_sums)
      self.activations.append(layer_activations)

  def __init_weights(self):
    """
    Initialisiert alle Gewichte zwischen den Layern so, wie sie
    durch den Parameter structure vorgegeben sind. Die Gewichte
    werden anfangs mit Zufallszahlen zwischen -1 und 1 vorbelegt.
    """
    self.weights = []

    for l in range(len(self.activations)):
      if l == 0:
        # Keine Gewichte ZUM Input-Layer. Wir hinterlegen hier ein leeres Array,
        # damit wir die Gewichtsmatrizen konsistent indizieren können und z. B.
        # die Gewichtsmatrix vom Input- zum ersten Hidden Layer den Index 1
        # statt 0 bekommt
        self.weights.append([])
      else:
        # Gewichte von Layer i-1 zu Layer i
        layer_weights = numpy.random.rand(
          len(self.activations[l]),    # Anzahl der Zeilen
          len(self.activations[l-1]),  # Anzahl der Spalten
        ) * 2 - 1 # Verschiebt die Zufallszahlen in den Bereich [-1, 1]

        # Füge diesen Satz an Gewichten zum Netz hinzu
        self.weights.append(layer_weights)

  def predict(self, X):
    # Füge eine Aktivierung von 1 für das Bias-Neuron
    input_activations = numpy.concatenate([numpy.array([1]), X])
    # Prüfe, ob die Anzahl der Input-Neuronen stimmt
    if len(self.activations[0]) != len(input_activations):
      raise Exception("Falsche Anzahl an Input-Werten übergeben")

    # Belege den Input-Layer mit den übergebenen Input plus Bias-Aktivierung
    self.activations[0] = input_activations
    
    for l in range(1, len(self.activations)):
      # Berechne die gewichteten Summen Z im aktuellen Layer aus dem Produkt
      # der Aktivierungen des vorherigen Layers und der Gewichte zwischen
      # beiden Layern
      self.weighted_sums[l] = numpy.matmul(self.weights[l], self.activations[l-1])

      # Berechne die Aktivierungen A
      if l == len(self.activations) - 1:
        # Output-Layer
        self.activations[l] = self.output_activation_func(self.weighted_sums[l])
      else:
        # Hidden Layer
        self.activations[l] = self.hidden_activation_func(self.weighted_sums[l])

    return self.activations[-1]

  def fit(self, training_data):
    # Lege Array für die partiellen Ableitungen (Deltas) der einzelnen Layer an
    partial_deltas = []
    # Obwohl wir die Deltas im Input-Layer nicht brauchen, legen wir trotzdem
    # auch dafür eine Liste an, damit wir die Indizes konsistent verwenden können
    for l in range(len(self.activations)):
      # Initialisiere die Deltas eines Layers mit so vielen Einträgen,
      # wie der Layer Neuronen hat. Befülle anfangs mit 0en
      partial_deltas.append(numpy.zeros((len(self.activations[l]), 1)))

    # Lege Array für die gewünschten Gewichtsanpassungen (delta W) an
    delta_W = []
    for l in range(len(self.activations)):
      if l == 0:
        # Obwohl wir die Deltas im Input-Layer nicht brauchen, legen wir trotzdem
        # auch dafür eine Liste an, damit wir die Indizes konsistent verwenden können
        delta_W.append([])
      else:
        # Output- und Hidden Layer:
        # Initialisiere die Deltas analog zu den Gewichten selbst
        # (siehe Funktion __init_weights__). Befülle aber diesmal anfangs
        # mit 0en statt mit Zufallszahlen
        delta_W.append(numpy.zeros((
          len(self.activations[l]), # Anzahl der Zeilen
          len(self.activations[l-1]), # Anzahl der Spalten
        )))

    # Wiederhole alles n_iterations-mal
    for _ in range(self.n_iterations):
      error = 0
      # Gehe durch alle Input-Output-Paare im Trainingsdatensatz
      for X, Y in training_data:
        # Prüfe, ob die Trainingsdaten-Outputs mit der Anzahl der Output-Neuronen
        # übereinstimmt
        if len(Y) != len(self.activations[-1]):
          raise Exception("Falsche Anzahl an erwarteten Output-Werten übergeben")

        prediction = self.predict(X)
        diff = prediction - Y
        output_layer_derivative = 2 * diff
        error += numpy.sum(diff ** 2)

        # Backpropagation: Rückwärts durch die Layer iterieren und deltas berechnen
        for l in range(len(self.activations)-1, 0, -1):
          if l == len(self.activations) - 1:
            # Output-Layer
            partial_deltas[l] = output_layer_derivative * \
              self.output_activation_func(self.weighted_sums[l], derivative=True)
          else:
            # Hidden Layer
            partial_deltas[l] = \
              numpy.matmul(self.weights[l+1].T, partial_deltas[l+1]) * \
              self.hidden_activation_func(self.weighted_sums[l], derivative=True)

          # Benutze die partiellen Deltas um die gewünschten Gewichtsanpassungen
          # delta W auszurechnen. Addiere sie pro Layer auf, später wird aus allen
          # delta Ws der einzelnen Trainingsbeispiele der Mittelwert genommen
          delta_W[l] += -self.eta * numpy.outer(partial_deltas[l], self.activations[l-1])

      # Lasse die Schleife bei 1 beginnen, weil zum Input-Layer keine Gewichte führen
      # und daher der Index 0 hier ignoriert werden muss
      for l in range(1, len(delta_W)):
        # Bilde die Mittelwerte der delta Ws, die von den einzelnen Trainingsbeispielen
        # ermittelt wurden
        self.weights[l] += delta_W[l] / len(training_data)
        # Setze die Delta Ws wieder auf 0 zurück für die nächste Iteration
        delta_W[l][:] = 0

      # Füge den akkumulierten Fehler zum Datensatz hinzu
      self.error_progression.append(error)

  def dump(self):
    print("Netzwerk-Architektur")
    print("--------")
    print("Layer:")
    for i in range(len(self.activations)):
      print("Layer", i)
      print(self.activations[i])
    print("--------")
    print("Gewichte:")
    for i in range(1, len(self.weights)):
      print("Gewichte von Layer", (i-1), "zu Layer", i)
      print(self.weights[i])

  def plot(self):
    plt.figure(1, figsize=(5, 5))
    plt.plot(self.error_progression)
    plt.xlabel('Iteration')
    plt.ylabel('Fehler')
    plt.show()

if __name__ == '__main__':
  nn = NeuralNetwork(
    structure = [3, 4, 1],
    eta=0.01,
    n_iterations=5000,
    output_activation_func=leaky_relu,
    hidden_activation_func=leaky_relu
  )

  training_data = [
    (numpy.array([0, 0, 0]), numpy.array([0])), # Tupel mit (Input, Output)
    (numpy.array([0, 0, 1]), numpy.array([1])),
    (numpy.array([0, 1, 0]), numpy.array([2])),
    (numpy.array([0, 1, 1]), numpy.array([3])),
    (numpy.array([1, 0, 0]), numpy.array([4])),
    # Die 5 lassen wir aus den Trainingsdaten weg und hoffen,
    # dass das Netz sie trotzdem "lernen" wird
    (numpy.array([1, 1, 0]), numpy.array([6])),
    (numpy.array([1, 1, 1]), numpy.array([7])),
  ]
  nn.fit(training_data)
  input = numpy.array([1, 0, 1]);
  prediction = nn.predict(input)
  nn.dump()
  print("=====")
  print("Prediction for unknown", input, ":", prediction)

  for X, _ in training_data:
    output = nn.predict(X)
    print("Prediction for known data", X, ":", output)

  nn.plot()

