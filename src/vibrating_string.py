'''
University: University of Amsterdam
Course: Scientific Computing
Authors: Margarita Petrova, Maan Scipio, Pjotr Piet
ID's: 15794717, 15899039, 12714933

Description:
This module contains a class for simulating the behavior of a vibrating string under different
initial conditions. It implements a finite difference scheme to solve the wave equation, allowing
for the simulation of string vibrations with various boundary and initial conditions. The user can
choose from three initial conditions:
1. sin(2*pi*x)
2. sin(5*pi*x)
3. sin(5*pi*x) if 1/5 < x < 2/5, 0 otherwise

The code also provides functions to animate the strings motion and plot time snapshots of the
string at various time steps.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import matplotlib.animation as animation

# Plotting parameters
sns.set(style="whitegrid")  # Use seaborn style
plt.rc('text', usetex=True)  # Disable LaTeX to avoid missing dependency issues
plt.rc('font', family='serif')
labelsize = 14
ticksize = 14
colors = sns.color_palette("Set2", 8)


class String:
    def __init__(self, N, L, initial_condition=1, c=1, dt=0.001):
        self.N = N
        self.L = L
        self.c = c
        self.dx = L / N
        self.dt = dt
        self.coefficient = (c * self.dt / self.dx) ** 2
        self.initial_condition = initial_condition

        # Discretized points
        self.x = np.linspace(0, L, N)

        self.t = 0
        # Empty wave function arrays at three time steps
        self.y_before = np.zeros(N)  # u(i, j-1)
        self.y = np.zeros(N)         # u(i, j)
        self.y_after = np.zeros(N)   # u(i, j+1)

        # Boundary conditions (fixed at zero)
        self.y[0] = 0
        self.y[-1] = 0

        # Initial conditions
        assert initial_condition in [1, 2, 3], "Invalid initial condition, choose between 1, 2, 3"

        if initial_condition == 1:
            self.y[:] = np.sin(2 * np.pi * self.x)
        elif initial_condition == 2:
            self.y[:] = np.sin(5 * np.pi * self.x)
        elif initial_condition == 3:
            for index, x in enumerate(self.x):
                if 1/5 < x < 2/5:
                    self.y[index] = np.sin(5 * np.pi * x)

        self.y_before[:] = self.y

    def step(self):
        """Computes the next time step using the finite difference method."""
        for i in range(1, self.N - 1):
            self.y_after[i] = (
                self.coefficient * (self.y[i + 1] - 2 * self.y[i] + self.y[i - 1])
                - self.y_before[i]
                + 2 * self.y[i]
            )

        # Update time step values
        self.y_before[:] = self.y
        self.y[:] = self.y_after

    def simulate(self, steps):
        """Runs the animation of the vibrating string."""
        fig, ax = plt.subplots()
        line, = ax.plot(self.x, self.y, color=colors[0])
        ax.set_ylim(-1, 1)
        plt.xlabel('x', fontsize=labelsize)
        plt.ylabel('y', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        ax.set_title(f"Initial Condition {self.initial_condition}", fontsize=labelsize)

        def update(frame):
            """Updates the plot at each frame."""
            self.step()  # Move one step forward
            line.set_ydata(self.y)  # Update plot
            return line,

        # Store the animation in `self.ani` to prevent garbage collection
        self.ani = animation.FuncAnimation(fig, update, frames=steps, interval=20, repeat=True)
        plt.show()

    def plot_snapshots(self):
        """Plots multiple time snapshots for each initial condition using seaborn."""
        N = 100
        L = 1.0
        dt = 0.001
        steps_per_snapshot = 300  # Steps before saving a snapshot
        num_snapshots = 5  # Number of times to plot

        fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=True, gridspec_kw={'wspace': 0})
        titles = [r'$\sin(2\pi x)$', r'$\sin(5\pi x)$', r'$\sin(5\pi x)$ if $1/5 < x < 2/5, 0$ otherwise']

        for condition in range(1, 4):  # Loop through initial conditions
            string = String(N=N, L=L, dt=dt, initial_condition=condition)
            ax = axes[condition - 1]
            for snap in range(num_snapshots):
                for _ in range(steps_per_snapshot):
                    string.step()
                sns.lineplot(x=string.x, y=string.y, ax=ax, label=f"t={snap * steps_per_snapshot * dt:.1f}s", color=colors[snap])

            ax.set_xlabel("x", fontsize=labelsize)

            ax.set_ylabel("y", fontsize=labelsize)
            ax.set_title(titles[condition - 1], fontsize=labelsize)
            if condition == 3:
                ax.legend(fontsize=9, loc='lower left')
            else:
                ax.legend().remove()

        plt.tight_layout()
        plt.savefig('../figures/string_snapshots.pdf')
        plt.show()


if __name__ == "__main__":
    # Run the original animation
    N = 50
    L = 1.0
    dt = 0.001
    diff = String(N=N, L=L, dt=dt, initial_condition=2)
    diff.simulate(steps=1000)

    # Run the additional snapshot plot
    diff.plot_snapshots()
