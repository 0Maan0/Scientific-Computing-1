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


# 1.2 The Time Dependent Diffusion Equation
class Twod:
    def __init__(self, N, L, dt=0.001, D=1):
        self.N = N
        self.L = L
        self.dx = L / N
        self.dt = dt
        self.D = D

        # assert to check stability
        assert (4 * self.D * self.dt / self.dx ** 2) <= 1, "Unstable: dt too large for given dx." + \
            " Max value for dt is: " + str(self.dx ** 2 / (4 * self.D))

        # discretized points
        self.x = np.linspace(0, L, N)
        self.y = np.linspace(0, L, N)
        self.t = 0

        # empty 2d array for the 2d time dependent thing
        self.c = np.zeros((N, N))

        # Set initial boundary conditions
        self.c[N-1, :] = 1  # top boundary
        self.c[0, :] = 0    # bottom boundary

    def check_boundary_conditions(self):
        """Check if boundary conditions are maintained"""
        assert np.allclose(self.c[self.N-1, :], 1), "Top boundary condition violated"
        assert np.allclose(self.c[0, :], 0), "Bottom boundary condition violated"

    def step(self):
        c_next = np.copy(self.c)

        c_next[self.N-1, :] = 1

        # Update all point using the diffusion equation as stated in the assignment
        for x in range(0, self.N):
            for y in range(1, self.N - 1):  # Skip boundary rows
                xmin1 = (x - 1) % self.N
                xplus1 = (x + 1) % self.N
                c_next[y, x] = self.c[y, x] + self.dt * self.D / self.dx ** 2 * \
                    (self.c[y, xplus1] + self.c[y, xmin1] + \
                     self.c[y + 1, x] + self.c[y - 1, x] - 4 * self.c[y, x])

        # Update array, time and check boundaries
        self.c = np.copy(c_next)
        self.t += self.dt
        self.check_boundary_conditions()

    def plot(self):
        """Plot the current state of the system as a 2D color map"""
        plt.figure(figsize=(8, 8))
        im = plt.imshow(self.c,
                       extent=[0, self.L, 0, self.L],
                       origin='lower',
                       cmap='viridis',
                       aspect='equal',
                       vmin=0, vmax=1)
        plt.colorbar(im, label='Concentration')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f't = {self.t:.3f}')
        plt.show()

    def animate(self, num_frames=200, interval=100, steps_per_frame=1):
        """Animate the evolution of the system
        Args:
            num_frames: Total number of animation frames
            interval: Time between frames in milliseconds
            steps_per_frame: Number of diffusion steps calculated per frame
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(self.c,
                    extent=[0, self.L, 0, self.L],
                    origin='lower',
                    cmap='viridis',
                    aspect='equal',
                    vmin=0, vmax=1)
        plt.colorbar(im, label='Concentration')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

        def update(frame):
            # Do multiple steps per frame
            for _ in range(steps_per_frame):
                self.step()
            im.set_array(self.c)
            ax.set_title(f't = {self.t:.3f}, frame = {frame * steps_per_frame}')
            return [im]

        anim = FuncAnimation(fig, update, frames=num_frames,
                        interval=interval, blit=False)
        plt.show()
        return anim

if __name__ == "__main__":
    # Example usage with stable parameters
    N = 50
    L = 1.0
    D = 1.0
    dx = L/N
    dt = 0.0001

    diff = Twod(N=N, L=L, dt=dt, D=D)

    # Run simulation with animation
    diff.animate(num_frames=1000, interval=1, steps_per_frame=5)
