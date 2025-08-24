import numpy
import matplotlib.pyplot as plt
from datetime import datetime
import random

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
            structure=None,
            eta=0.01,
            n_iterations=1000,
            output_activation_func=relu,
            hidden_activation_func=relu,
            batch_size=0
    ):
        self.structure = structure
        self.eta = eta
        self.n_iterations = n_iterations
        self.output_activation_func = output_activation_func
        self.hidden_activation_func = hidden_activation_func
        self.batch_size = batch_size
        self.error_progression = []
        
        if isinstance(self.structure, list):
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

    def save_to_file(self, filename):
        numpy.savez(filename, *self.weights)

    def load_from_file(self, filename):
        loaded_weights = numpy.load(filename)
        self.weights = [loaded_weights[key] for key in loaded_weights]
        self.structure = []
        
        for l in range(1, len(self.weights)):
            self.structure.append(self.weights[l].shape[1] - 1)
        
        self.structure.append(self.weights[-1].shape[0])
        self.__init_layers()

    def predict(self, X):
        if len(X) != len(self.activations[0]) - 1:
            raise Exception("Falsche Anzahl an Input-Werten übergeben")

        self.activations[0][1:] = X
        
        for l in range(1, len(self.activations)):
            #self.activations[l-1] = numpy.clip(self.activations[l-1], -1e3, 1e3)
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
                delta_W.append(numpy.zeros(self.weights[l].shape))

        for i in range(self.n_iterations):
            # now = datetime.now()
            # print("Starting iteration", i, "at", now.strftime("%H:%M:%S"))
            training_batch = (
                training_data
                if self.batch_size == 0
                else random.sample(training_data, self.batch_size)
            )
            error = 0
            for X, Y in training_batch:
                if len(Y) != len(self.activations[-1]):
                    raise Exception("Falsche Anzahl an erwarteten Output-Werten übergeben")

                prediction = self.predict(X)
                diff = prediction - Y
                output_layer_derivative = 2 * diff
                error += numpy.sum(diff ** 2)

                for l in range(len(self.activations)-1, 0, -1):
                    if l == len(self.activations) - 1:
                        if self.output_activation_func == softmax:
                            partial_deltas[l] = diff
                        else:
                            partial_deltas[l] = output_layer_derivative * \
                                self.output_activation_func(self.weighted_sums[l], derivative=True)
                    else:
                        partial_deltas[l] = \
                            numpy.matmul(self.weights[l+1][:, 1:].T, partial_deltas[l+1]) * \
                            self.hidden_activation_func(self.weighted_sums[l], derivative=True)

                    delta_W[l] += -self.eta * numpy.matmul(partial_deltas[l], self.activations[l-1].T)

            for l in range(1, len(delta_W)):
                self.weights[l] += delta_W[l] / len(training_batch)
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
