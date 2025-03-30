import numpy
import matplotlib.pyplot as plt

def relu(z, derivative=False):
  if derivative:
    return numpy.heaviside(z, 1)
  return numpy.maximum(z, 0)

def leaky_relu(z, derivative=False):
  if derivative:
    return numpy.where(z > 0, 1, 0.1)
  return numpy.where(z > 0, z, 0.1 * z)

def sigmoid(z, derivative=False):
  if derivative:
    return sigmoid(z) * (1 - sigmoid(z))
  return 1 / (1 + numpy.exp(-z))

def identity(z, derivative=False):
  if derivative:
    return numpy.ones_like(z)
  return z

def softmax(z):
  stabilized_z = z - numpy.max(z)
  exp_z = numpy.exp(stabilized_z)
  
  return exp_z / numpy.sum(exp_z)
  

class NeuralNetwork():
  def __init__(
      self,
      structure,
      eta=0.01,
      n_iterations=1000,
      output_activation_func=relu,
      hidden_activation_func=relu,
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
    self.activations = []
    self.weighted_sums = []
    
    for l in range(len(self.structure)):
      is_output_layer = l == len(self.structure) - 1
      n_activations = self.structure[l] if is_output_layer else self.structure[l] + 1
      n_weighted_sums = self.structure[l]
      layer_weighted_sums = numpy.zeros((n_weighted_sums, 1))
      layer_activations = numpy.ones((n_activations, 1))

      self.weighted_sums.append(layer_weighted_sums)
      self.activations.append(layer_activations)

  def __init_weights(self):
    self.weights = []

    for l in range(len(self.activations)):
      if l == 0:
        self.weights.append([])
      else:
        layer_weights = numpy.random.rand(
          len(self.weighted_sums[l]),
          len(self.activations[l-1]),
        ) * 2 - 1

        self.weights.append(layer_weights)

  def predict(self, X):
    if len(X) != len(self.activations[0]) - 1:
      raise Exception("Falsche Anzahl an Input-Werten übergeben")

    self.activations[0][1:] = X
    
    for l in range(1, len(self.activations)):
      self.weighted_sums[l] = numpy.matmul(self.weights[l], self.activations[l-1])

      if l == len(self.activations) - 1:
        self.activations[l] = self.output_activation_func(self.weighted_sums[l])
      else:
        self.activations[l][1:] = self.hidden_activation_func(self.weighted_sums[l])

    return self.activations[-1]

  def train(self, training_data):
    partial_deltas = []
    for l in range(len(self.activations)):
      partial_deltas.append(numpy.zeros((len(self.weighted_sums[l]), 1)))

    delta_W = []
    for l in range(len(self.activations)):
      if l == 0:
        delta_W.append([])
      else:
        delta_W.append(numpy.zeros((
          len(self.weighted_sums[l]),
          len(self.activations[l-1]),
        )))

    for _ in range(self.n_iterations):
      error = 0
      for X, Y in training_data:
        if len(Y) != len(self.activations[-1]):
          raise Exception("Falsche Anzahl an erwarteten Output-Werten übergeben")

        prediction = self.predict(X)
        diff = prediction - Y
        output_layer_derivative = 2 * diff
        error += numpy.sum(diff ** 2)

        for l in range(len(self.activations)-1, 0, -1):
          if l == len(self.activations) - 1:
            partial_deltas[l] = output_layer_derivative * \
              self.output_activation_func(self.weighted_sums[l], derivative=True)
          else:
            partial_deltas[l] = \
              numpy.matmul(self.weights[l+1][:, 1:].T, partial_deltas[l+1]) * \
              self.hidden_activation_func(self.weighted_sums[l], derivative=True)

          delta_W[l] += -self.eta * numpy.matmul(partial_deltas[l], self.activations[l-1].T)

      for l in range(1, len(delta_W)):
        self.weights[l] += delta_W[l] / len(training_data)
        delta_W[l][:] = 0

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
    structure=[3, 4, 1],
    eta=0.01,
    n_iterations=3000,
    output_activation_func=leaky_relu,
    hidden_activation_func=leaky_relu
  )

  training_data = [
    (numpy.array([[0], [0], [0]]), numpy.array([[0]])),
    (numpy.array([[0], [0], [1]]), numpy.array([[1]])),
    (numpy.array([[0], [1], [0]]), numpy.array([[2]])),
    (numpy.array([[0], [1], [1]]), numpy.array([[3]])),
    (numpy.array([[1], [0], [0]]), numpy.array([[4]])),
    #(numpy.array([[1], [0], [1]]), numpy.array([[5]])),
    (numpy.array([[1], [1], [0]]), numpy.array([[6]])),
    (numpy.array([[1], [1], [1]]), numpy.array([[7]])),
  ]
  nn.train(training_data)
  input = numpy.array([[1], [0], [1]]);
  prediction = nn.predict(input)
  #nn.dump()
  print("Vorhersage für unbekannten Input", input, ":", prediction[0, 0])

  for X, _ in training_data:
    output = nn.predict(X)
    print("Vorhersage für bekannten Input", X, ":", output[0, 0])

  nn.plot()

