import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Perceptron():
    def __init__(self, w0, w1, b, eta):
        self.w0 = w0
        self.w1 = w1
        self.b = b
        self.eta = eta
        self.total_num_iterations = 0
        self.history = []  # Speichert Gewichtshistorie für die Animation

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

            self.history.append((self.b, self.w0, self.w1))


def animate_perceptron(pcp, training_data):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Punkte plotten
    ax.scatter(*zip(*training_data[0][1]), color='blue', label='Klasse 0')
    ax.scatter(*zip(*training_data[1][1]), color='red', label='Klasse 1')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-4, 10)
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.grid()
    ax.legend()
    line, = ax.plot([], [], color='green')

    def update(frame):
        b, w0, w1 = pcp.history[frame]
        m = -(w0 / w1)
        n = -(b / w1)
        x_vals = [-10, 10]
        y_vals = [m * x + n for x in x_vals]
        
        line.set_data(x_vals, y_vals)
        ax.set_title(f"Iteration {frame+1}")
        return line,

    _ = FuncAnimation(fig, update, frames=len(pcp.history), interval=50, repeat=False)
    plt.show()

if __name__ == '__main__':
    classified_points = [
        (0, [(2, 0), (-4, 3), (-5, 5), (-2, 2), (-3, 1), (-1, -1)]),
        (1, [(0, 4), (2, 5),  (5, 3), (3, 2), (1, 6), (7, 1)])
    ]

    pcp = Perceptron(0.5, -0.3, 0.2, 0.1)
    pcp.train(classified_points)

    print("Final weights: (b, w0, w1) = ", (pcp.b, pcp.w0, pcp.w1))
    print("Total # of iterations:", pcp.total_num_iterations)

    animate_perceptron(pcp, classified_points)
