import numpy as np
import matplotlib.pyplot as plt
import time

class String:
    def __init__(self, N, L, c=1, dt=0.001):
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
        self.y[:] = np.sin(2* np.pi * self.x) #np.exp(-100 * (self.x - L / 2) ** 2)
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

    def simulate(self, steps, plot_interval=1):
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot(self.x, self.y, color = 'tab:blue')
        ax.set_ylim(-1, 1)
        plt.xlabel('x')
        plt.ylabel('y')

        for step in range(steps):
            self.step()
            if step % plot_interval == 0:
                line.set_ydata(self.y)
                plt.pause(0.01)
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    wave = String(N=100, L=1.0)
    wave.simulate(steps=500, plot_interval=10)