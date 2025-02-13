import numpy as np  
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class time_independent_diffusion:
    def __init__(self, N, L, epsilon = 1e-5, max_iter = 100):
        self.N = N
        self.L = L
        self.dx = L / N
        self.epsilon = epsilon
        self.max_iter = max_iter

        # discretized points
        self.x = np.linspace(0, L, N)
        self.y = np.linspace(0, L, N)

        # empty 2d array for the 2d time independent thing
        self.c = np.zeros((N, N))

        # Set initial boundary conditions
        self.c[N-1, :] = 1
        self.c[0, :] = 0

    def check_boundary_conditions(self):
        """Check if boundary conditions are maintained"""
        assert np.allclose(self.c[self.N-1, :], 1), "Top boundary condition violated"
        assert np.allclose(self.c[0, :], 0), "Bottom boundary condition violated"
    
    def step(self):
        c_next = np.copy(self.c)
        c_next[self.N-1, :] = 1
        iteration = 0
        while iteration < self.max_iter:
            for x in range(1, self.N-1): # hier range(0, self.N)?
                for y in range(1, self.N-1):
                    xmin1 = (x - 1) % self.N
                    xplus1 = (x + 1) % self.N
                    c_next[x, y] = 0.25 * (self.c[xplus1, y] + self.c[xmin1, y] + 
                                            self.c[x, y+1] + self.c[x, y-1])
            delta = np.max(np.abs(c_next - self.c))
            if delta < self.epsilon:
                print(f"Converged after {iteration} iterations (delta={delta:.2e}).")
                break
            iteration += 1

        # Update array, time and check boundaries
        self.c = np.copy(c_next)
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
        #plt.title(f't = {self.t:.3f}')
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
            #ax.set_title(f't = {self.t:.3f}, frame = {frame * steps_per_frame}')
            return [im]

        anim = FuncAnimation(fig, update, frames=num_frames,
                        interval=interval, blit=False)
        plt.show()
        return anim


if __name__ == "__main__":
    # Example usage with stable parameters
    N = 50
    L = 1.0

    diff = time_independent_diffusion(N=N, L=L)

    # Run simulation with animation
    diff.animate(num_frames=1000, interval=1, steps_per_frame=10)
