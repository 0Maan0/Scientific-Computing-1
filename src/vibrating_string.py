import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


# 1.2 The Time Dependent Diffusion Equation
class Twod:
    def __init__(self, N, L, dt=0.001, D=1):
        self.N = N
        self.L = L
        self.dx = L / N
        self.dt = dt
        self.D = D

        # assert to check stability
        assert (4 * self.dt  / self.dx ** 2) > 1, "Unstable"

        # discretized points
        self.x = np.linspace(0, L, N)
        self.y = np.linspace(0, L, N)
        self.t = 0

        # empty 2d array for the 2d time dependent thing
        self.c = np.zeros((N, N))

        # boundary starting conditions: c(x, y=1;t) = 1 and c(x, y=0;t) = 0
        for i in range(0, N):
            self.c[N-1, i] = 1

        # copy for the next time step
        self.c_next = np.copy(self.c)

    def step(self):
        for x in range(0, self.N):
            for y in range(1, self.N - 1):
                xmin1 = (x - 1) % self.N
                xplus1 = (x + 1) % self.N
                # Fixed indexing: self.c[y, x] instead of self.c[x, y]
                self.c_next[y, x] = self.c[y, x] + self.dt * self.D / self.dx ** 2 * \
                    (self.c[y, xplus1] + self.c[y, xmin1] + self.c[y + 1, x] + self.c[y - 1, x] - 4 * self.c[y, x])

        # update time step values
        self.c = np.copy(self.c_next)  # Added np.copy() to prevent reference issues

        self.t += self.dt

    def plot_3d(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(self.x, self.y)
        ax.plot_surface(X, Y, self.c, cmap='viridis')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Concentration')
        ax.set_title(f"Time Step: {self.t:.3f}")

        plt.show()

    def animate(self, num_frames=200, interval=50):
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(self.x, self.y)

        def init():
            ax.clear()
            return [ax.pcolormesh(X, Y, self.c, cmap='viridis')]

        def update(frame):
            self.step()
            ax.clear()
            mesh = ax.pcolormesh(X, Y, self.c, cmap='viridis')
            ax.set_title(f'Time: {self.t:.3f}')
            return [mesh]

        anim = FuncAnimation(fig, update, frames=num_frames,
                           init_func=init, interval=interval,
                           blit=True)
        plt.colorbar(ax.pcolormesh(X, Y, self.c, cmap='viridis'))
        plt.show()


if __name__ == "__main__":
    # Example usage:
    diff = Twod(N=50, L=1.0)
    diff.animate(num_frames=200, interval=100)

    # wave = String(N=100, L=1.0, initial_condition= 3)
    # wave.simulate(steps=500)
