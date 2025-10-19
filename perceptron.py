import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, w0, w1, b, eta):
        self.w0 = w0
        self.w1 = w1
        self.b = b
        self.eta = eta
        self.total_num_iterations = 0

    def train(self, training_data):
        total_num_points = len(training_data[0][1]) + len(training_data[1][1])

        while True:
            d_b = 0
            d_w0 = 0
            d_w1 = 0
            total_error = 0;
            self.total_num_iterations += 1

            for (y, points) in training_data:
                for (x_0, x_1) in points:
                    z = self.w0 * x_0 + self.w1 * x_1 + self.b
                    a = 1 if z > 0 else 0
                    error = y - a
                    d_w0 += self.eta * error * x_0
                    d_w1 += self.eta * error * x_1
                    d_b += self.eta * error
                    total_error += error ** 2

            if total_error == 0:
                return

            if self.total_num_iterations >= 100000:
                print(f"Giving up after {self.total_num_iterations} iterations.")
                return

            d_w0 = d_w0 / total_num_points
            d_w1 = d_w1 / total_num_points
            d_b = d_b / total_num_points

            self.w0 += d_w0
            self.w1 += d_w1
            self.b += d_b


if __name__ == '__main__':
    classified_points = [
        (0, [(2, 0), (-4, 3), (-5, 5), (-2, 2), (-3, 1), (-1, -1)]),
        (1, [(0, 4), (2, 5),  (5, 3), (3, 2), (1, 6), (7, 1)])
    ]

    pcp = Perceptron(0.5, -0.3, 0.2, 0.1)
    pcp.train(classified_points)
    
    m = -(pcp.w0 / pcp.w1)
    n = -(pcp.b / pcp.w1)

    print(f"Final weights: (b, w0, w1) = {(pcp.b, pcp.w0, pcp.w1)}")
    print(f"Total # of iterations: {pcp.total_num_iterations}")
    print(f"Linear function params (m, n) = {(m, n)}")

    plt.figure(1, figsize=(5, 5))
    plt.scatter(*zip(*classified_points[0][1]), color='blue', label='Klasse 0')
    plt.scatter(*zip(*classified_points[1][1]), color='red', label='Klasse 1')
    plt.legend()

    X = [-10, 10]
    Y = [m * x + n for x in X]
    plt.plot(X, Y, color='green')
    
    plt.xlabel('x_0')
    plt.ylabel('x_1')
    plt.grid()
    plt.show()

