import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class time_independent_diffusion:
    def __init__(self, N, L, epsilon = 1e-5, max_iter = 10000, method='jacobi', omega=1.0):
        self.N = N
        self.L = L
        self.dx = L / N
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.method = method
        self.omega = omega

        # discretized points
        self.x = np.linspace(0, L, N)
        self.y = np.linspace(0, L, N)

        # two grids for Jacobi method
        self.c = np.zeros((N, N))

        # Set initial boundary conditions
        self.c[N-1, :] = 1
        self.c[0, :] = 0

    def check_boundary_conditions(self):
        """Check if boundary conditions are maintained"""
        assert np.allclose(self.c[self.N-1, :], 1), "Top boundary condition violated"
        assert np.allclose(self.c[0, :], 0), "Bottom boundary condition violated"

    def jacobi_step(self):
        """
        One Jacobi iteration step.
        """
        c_next = np.copy(self.c)
        c_next[self.N-1, :] = 1 # top boundary
        c_next[0, :] = 0 # bottom boundary

        # Vectorized update with wrapping boundary conditions (excluding first and last row)
        c_next[1:-1, :] = 0.25 * (
            self.c[2:, :] +     # below (next row, non-wrapped)
            self.c[:-2, :] +    # above (previous row, non-wrapped)
            self.c[1:-1, (np.arange(self.c.shape[1]) + 1) % self.c.shape[1]] +  # right neighbor (wraps around)
            self.c[1:-1, (np.arange(self.c.shape[1]) - 1) % self.c.shape[1]]    # left neighbor (wraps around)
        )

        # maximum change
        delta = np.max(np.abs(c_next - self.c))

        # Update grid
        self.c = c_next

        return delta

    def gauss_seidel_step(self):
        """
        One Gauss-Seidel iteration step.
        """
        c_old = np.copy(self.c)

        # boundary conditions
        self.c[self.N-1, :] = 1  # top boundary
        self.c[0, :] = 0  # bottom boundary

        # Update all point using the diffusion equation as stated in the assignment
        for x in range(0, self.N):
            for y in range(1, self.N - 1):  # Skip boundary rows
                xmin1 = (x - 1) % self.N
                xplus1 = (x + 1) % self.N
                self.c[y, x] = 1 / 4 * (self.c[y, xplus1] + self.c[y, xmin1] + self.c[y + 1, x] + \
                                        self.c[y - 1, x])

        # maximum change
        delta = np.max(np.abs(self.c - c_old))

        return delta

    def sor_step(self):
        """
        One Successive Over-Relaxation (SOR) iteration step.
        """
        c_old = np.copy(self.c)

        # boundary conditions
        self.c[self.N-1, :] = 1  # top boundary
        self.c[0, :] = 0  # bottom boundary

        # Update all point using the diffusion equation as stated in the assignment
        for x in range(0, self.N):
            for y in range(1, self.N - 1):  # Skip boundary rows
                xmin1 = (x - 1) % self.N
                xplus1 = (x + 1) % self.N
                self.c[y, x] = self.omega / 4 * (self.c[y, xplus1] + self.c[y, xmin1] + self.c[y + 1, x] + \
                                        self.c[y - 1, x]) + (1 - self.omega) * self.c[y, x]


        # Calculate maximum change
        delta = np.max(np.abs(self.c - c_old))

        return delta


    def solve(self):
        """
        Solve the time-independent diffusion equation using Jacobi iteration.
        """
        # history tracking
        self.delta_history = []
        self.iterations = None

        # Select the appropriate method
        if self.method == 'jacobi':
            step_method = self.jacobi_step
        elif self.method == 'gauss-seidel':
            step_method = self.gauss_seidel_step
        elif self.method == 'sor':
            step_method = self.sor_step
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # iteration
        for iteration in range(self.max_iter):
            delta = step_method()
            self.delta_history.append(delta)

            if delta < self.epsilon:
                print(f"Converged after {iteration+1} iterations (delta={delta:.2e})")
                self.iterations = iteration + 1
                break

        if self.iterations is None:
            print(f"Warning: Failed to converge after {self.max_iter} iterations")
            print(f"Last delta: {delta}")
            print(f"Epsilon: {self.epsilon}")

        return self.iterations

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
        plt.title(f'Steady State Concentration Distribution({self.method.capitalize()})')
        plt.tight_layout()
        plt.show()

    def plot_convergence(self):
        """
        Plot the convergence history (delta vs iterations).
        """
        plt.figure(figsize=(10, 6))
        plt.semilogy(self.delta_history)
        plt.title('Convergence of Jacobi Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Maximum Change (log scale)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_all_convergence(self):
        """
        Plot the convergence history (delta vs iterations) for all methods.
        """
        plt.plot(figsize=(18, 6))
        colors = ['tab:blue', 'tab:green', 'tab:red']
        omega_values = [1.0, 1.0, 1.5]

        for i, method in enumerate(['jacobi', 'gauss-seidel', 'sor']):
            self.omega = omega_values[i]
            diff = time_independent_diffusion(N=self.N, L=self.L, epsilon=self.epsilon,
                                              method=method, omega=self.omega)
            diff.solve()
            plt.semilogy(diff.delta_history, label=method.capitalize(), color=colors[i])
        plt.title(f'Conergence of different Methods')
        plt.xlabel('Iteration')
        plt.ylabel('Maximum Change (log scale)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    # def animate(self, num_frames=200, interval=100, steps_per_frame=1):
    #     """Animate the evolution of the system
    #     Args:
    #         num_frames: Total number of animation frames
    #         interval: Time between frames in milliseconds
    #         steps_per_frame: Number of diffusion steps calculated per frame
    #     """
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     im = ax.imshow(self.c,
    #                 extent=[0, self.L, 0, self.L],
    #                 origin='lower',
    #                 cmap='viridis',
    #                 aspect='equal',
    #                 vmin=0, vmax=1)
    #     plt.colorbar(im, label='Concentration')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')

    #     def update(frame):
    #         # Do multiple steps per frame
    #         for _ in range(steps_per_frame):
    #             self.step()
    #         im.set_array(self.c)
    #         #ax.set_title(f't = {self.t:.3f}, frame = {frame * steps_per_frame}')
    #         return [im]

    #     anim = FuncAnimation(fig, update, frames=num_frames,
    #                     interval=interval, blit=False)
    #     plt.show()
    #     return anim


if __name__ == "__main__":
    # Example usage with stable parameters
    N = 50
    L = 1.0
    epsilon = 1e-6

    methods = [
        ('jacobi', 1.0),
        ('gauss-seidel', 1.0),
        ('sor', 1.7),
        ('sor', 1.8),
        ('sor', 1.9),
    ]

    diff = time_independent_diffusion(N=N, L=L, epsilon=epsilon, method='jacobi')
    diff.plot_all_convergence()

    # for method, omega in methods:
    #     print(f"\nSolving with {method.upper()} method:")
    #     diff = time_independent_diffusion(N=N, L=L, epsilon=epsilon,
    #                                       method=method, omega=omega)
    #     diff.solve()
    #     diff.plot()
    #     diff.plot_convergence()
