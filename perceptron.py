import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self):
        self.w0 = 0
        self.w1 = 0
        self.b = 0
        self.total_num_iterations = 0

    def train(self, training_data):
        total_num_points = len(training_data[0][1]) + len(training_data[1][1])
        max_num_iterations = 100_000

        for i in range(max_num_iterations):
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
                    d_w0 += error * x_0
                    d_w1 += error * x_1
                    d_b += error
                    total_error += error ** 2

            if total_error == 0:
                return

            d_w0 = d_w0 / total_num_points
            d_w1 = d_w1 / total_num_points
            d_b = d_b / total_num_points

            self.w0 += d_w0
            self.w1 += d_w1
            self.b += d_b
            
        print(f"Giving up after {max_num_iterations} iterations.")


if __name__ == '__main__':
    classified_points = [
        (0, [(0.54, 0.31), (0.37, 0.48), (0.46, 0.42), (0.56, 0.25), (0.77, 0.22), (0.42, 0.46)]),
        (1, [(0.54, 0.60), (0.71, 0.62), (0.51, 0.63), (0.67, 0.44), (0.37, 0.81), (0.65, 0.54)])
    ]

    pcp = Perceptron()
    pcp.train(classified_points)
    
    m = -(pcp.w0 / pcp.w1)
    n = -(pcp.b / pcp.w1)

    print(f"Final weights: {pcp.b=}, {pcp.w0=}, {pcp.w1=}")
    print(f"Total # of iterations: {pcp.total_num_iterations}")
    print(f"Linear function params: {m=}, {n=}")

    plt.scatter(*zip(*classified_points[0][1]), color='blue', label='Iris Setosa')
    plt.scatter(*zip(*classified_points[1][1]), color='red', label='Iris Versicolor')
    plt.legend()

    X = [0, 1]
    Y = [m * x + n for x in X]
    plt.plot(X, Y, color='green')
    
    plt.xlabel('x_0')
    plt.ylabel('x_1')
    plt.grid()
    plt.tight_layout()
    plt.show()

