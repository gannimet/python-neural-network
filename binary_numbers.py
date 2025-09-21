from neural_network import NeuralNetwork, leaky_relu, sigmoid
import numpy

if __name__ == '__main__':
    nn = NeuralNetwork(
        structure=[3, 4, 1],
        eta=0.01,
        n_iterations=5_000,
        output_activation_func=leaky_relu,
        hidden_activation_func=leaky_relu
    )
    
    numpy.set_printoptions(suppress=True)

    # Trainingsdaten für Regression
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
    print("Vorhersage für unbekannten Input", input, ":", prediction)

    for X, _ in training_data:
        output = nn.predict(X)
        print("Vorhersage für bekannten Input", X, ":", output)

    nn.plot()