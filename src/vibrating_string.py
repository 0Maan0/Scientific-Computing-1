import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation

# 1.1 Vibrating string
class String:
    def __init__(self, N, L, initial_condition = 1, c=1, dt=0.001):
        self.N = N
        self.L = L
        self.c = c
        self.dx = L / N
        self.dt =  dt
        self.coefficient = (c * self.dt/self.dx) ** 2

        # discretized points
        self.x = np.linspace(0, L, N)

        self.t = 0
        # empty wave function arrays at three time steps
        self.y_before = np.zeros(N)  # u(i, j-1)
        self.y = np.zeros(N)       # u(i, j)
        self.y_after = np.zeros(N)  # u(i, j+1)

        # boundary conditions 0 on start and end
        self.y[0] = 0
        self.y[-1] = 0

        # initial conditions ==> start with a gaussian curve eg. a wave
        assert initial_condition in [1, 2, 3], "Invalid initial condition"
        if initial_condition == 1:
            self.y[:] = np.sin(2* np.pi *self.x)
        elif initial_condition == 2:
            self.y[:] = np.sin(5* np.pi * self.x)
        elif initial_condition == 3:
            for index, x in enumerate(self.x):
                if x > 1/5 and x < 2/5:
                    self.y[index] = np.sin(5* np.pi * x)
                else:
                    self.y[index] = 0
        self.y_before[:] = self.y

    def step(self):
        for i in range(1, self.N - 1):
            self.y_after[i] = (
                self.coefficient * (self.y[i + 1] - 2 * self.y[i] + self.y[i - 1])
                - self.y_before[i]
                + 2 * self.y[i]
            )

        # update time step values
        self.y_before[:] = self.y
        self.y[:] = self.y_after

    def simulate(self, steps):

        fig, ax = plt.subplots()
        line, = ax.plot(self.x, self.y, color = 'tab:blue')
        ax.set_ylim(-1, 1)
        plt.xlabel('x')
        plt.ylabel('y')

        def update(frame):
            for step in range(steps):
                self.step()
                line.set_ydata(self.y)
                return line
        ani = animation.FuncAnimation(fig, func=update, frames=steps, interval=20, repeat=False)
        ani.save(filename="ffmpeg_example.mkv", writer="ffmpeg")
        plt.show()


if __name__ == "__main__":
    wave = String(N=100, L=1.0, initial_condition=3)
    wave.simulate(steps=50)
    wave.simulate(steps=500)
