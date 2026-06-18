import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Perceptron():
    def __init__(self):
        self.w0 = 0
        self.w1 = 0
        self.b = 0
        self.total_num_iterations = 0
        self.history = []

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
            self.history.append((self.b, self.w0, self.w1))
            
        print(f"Giving up after {max_num_iterations} iterations.")



def animate_perceptron(pcp, training_data):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Punkte plotten
    ax.scatter(*zip(*training_data[0][1]), color='blue', label='Klasse 0')
    ax.scatter(*zip(*training_data[1][1]), color='red', label='Klasse 1')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x_0')
    ax.set_ylabel('x_1')
    ax.grid()
    ax.legend()
    line, = ax.plot([], [], color='green')

    def update(frame):
        b, w0, w1 = pcp.history[frame]
        m = -(w0 / w1)
        n = -(b / w1)
        x_vals = [0, 1]
        y_vals = [m * x + n for x in x_vals]
        
        line.set_data(x_vals, y_vals)
        ax.set_title(f"Iteration {frame+1}")
        return line,

    _ = FuncAnimation(fig, update, frames=len(pcp.history), interval=100, repeat=False)
    plt.show()

if __name__ == '__main__':
    classified_points = [
        (0, [(0.54, 0.31), (0.37, 0.48), (0.46, 0.42), (0.56, 0.25), (0.77, 0.22), (0.42, 0.46)]),
        (1, [(0.54, 0.60), (0.71, 0.62), (0.51, 0.63), (0.67, 0.44), (0.37, 0.81), (0.65, 0.54)])
    ]

    pcp = Perceptron()
    pcp.train(classified_points)

    print("Final weights: (b, w0, w1) = ", (pcp.b, pcp.w0, pcp.w1))
    print("Total # of iterations:", pcp.total_num_iterations)

    animate_perceptron(pcp, classified_points)
