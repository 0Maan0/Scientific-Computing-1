'''
This modulle contains a class for simulating a vibrating string. 
You can choose between three initial conditions for the string.
1. sin(2*pi*x)
2. sin(5*pi*x)
3. sin(5*pi*x) if x is between 1/5 and 2/5, 0 otherwise
'''
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from scipy.special import erfc
import seaborn as sns
import matplotlib.animation as animation

# Plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
labelsize = 14
ticksize = 14
colors = sns.color_palette("Set2", 8)


# 1.1 Vibrating string
class String:
    def __init__(self, N, L, initial_condition = 1, c=1, dt=0.001):
        self.N = N
        self.L = L
        self.c = c
        self.dx = L / N
        self.dt =  dt
        self.coefficient = (c * self.dt/self.dx) ** 2
        self.initial_condition = initial_condition

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
        assert initial_condition in [1, 2, 3], "Invalid initial condition, choose between 1, 2, 3"

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

        self.initialcondition_string = ["$\sin(2\pi x)$", "$\sin(5\pi x)$", "$\sin(5\pi x) $ if $ 1/5 < x < 2/5, 0 $ otherwise"]

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
        line, = ax.plot(self.x, self.y, color = colors[0])
        ax.set_ylim(-1, 1)
        plt.xlabel('x', fontsize=labelsize)
        plt.ylabel('y', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.title(rf'$\Psi(x, t=0) = ${self.initialcondition_string[self.initial_condition-1]}', fontsize=labelsize)

        def update(frame):
            for step in range(steps):
                self.step()
                line.set_ydata(self.y)
                return line
        ani = animation.FuncAnimation(fig, func=update, frames=steps, interval=20, repeat=False)
        ani.save(filename=f"../figures/string_animation_ic{self.initial_condition}.mkv", writer="ffmpeg")
        plt.show()
if __name__ == "__main__":
    # Example usage with stable parameters
    N = 50
    L = 1.0
    dt = 0.001

    diff = String(N=N, L=L, dt=dt, initial_condition=2)

    # Run simulation with animation
    diff.simulate(steps=1000)
    
    for _ in range(1000):
        diff.step()
