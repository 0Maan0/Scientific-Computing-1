import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation


# 1.2 The Time Dependent Diffusion Equation
class Diffusion:
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
        """
        Update the system by one time step using the diffusion equation. This
        is the 2D differential equation for diffusion.
        """
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

    diff = Diffusion(N=N, L=L, dt=dt, D=D)

    # Run simulation with animation
    diff.animate(num_frames=1000, interval=1, steps_per_frame=10)
