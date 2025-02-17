import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
titlesize = 16
labelsize = 14
colors = sns.color_palette("Set2", 8)
ticksize = 14

class time_independent_diffusion:
    def __init__(self, N, L, epsilon = 1e-5, max_iter = 10000, method='Jacobi', omega=1.0):
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
        if self.method == 'Jacobi':
            step_method = self.jacobi_step
        elif self.method == 'Gauss-Seidel':
            step_method = self.gauss_seidel_step
        elif self.method == 'SOR':
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
        plt.xlabel('x', fontsize=labelsize)
        plt.ylabel('y', fontsize=labelsize)
        plt.title(f'Steady State Concentration Distribution({self.method.capitalize()})', fontsize=titlesize)
        plt.yticks(fontsize=ticksize)
        plt.xticks(fontsize=ticksize)
        plt.tight_layout()
        plt.show()

    def plot_convergence(self):
        """
        Plot the convergence history (delta vs iterations).
        """
        plt.figure(figsize=(8, 8))
        plt.semilogy(self.delta_history)
        plt.title('Convergence of Jacobi Iteration', fontsize=titlesize)
        plt.xlabel('Iteration', fontsize=labelsize)
        plt.ylabel('Maximum Change (log scale)', fontsize=labelsize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_all_convergence(self):
        """
        Plot the convergence history (delta vs iterations) for all methods.
        """
        plt.plot(figsize=(8, 8))
        omega_values = [1.0, 1.0, 1.7, 1.8, 1.9]

        for i, method in enumerate(['Jacobi', 'Gauss-Seidel', 'SOR', 'SOR', 'SOR']):
            omega_temp = omega_values[i]
            diff = time_independent_diffusion(N=self.N, L=self.L, epsilon=self.epsilon,
                                              method=method, omega=omega_temp)
            diff.solve()
            plt.semilogy(diff.delta_history, label=rf"{method.capitalize()}, $\omega$ = {omega_temp}", color=colors[i])
        plt.title(f'Convergence of different Methods', fontsize=titlesize)
        plt.xlabel('Iteration', fontsize=labelsize)
        plt.ylabel(r'Maximum Change $\delta$ (log scale)', fontsize=labelsize)
        plt.grid(True)
        plt.legend(fontsize=ticksize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.tight_layout()
        plt.savefig('../figures/convergence.pdf')
        plt.show()
    def plot_all_concentrations(self):
        # plot c for each value of y for each method
        methods = ['Jacobi', 'Gauss-Seidel', 'SOR']
        print(f"Plotting concentration at each y for different methods")
        plt.figure(figsize=(8, 8))
        linestyles = ['-', '--', '-.']
        for i, method in enumerate(methods):
            diff = time_independent_diffusion(N=self.N, L=self.L, epsilon=self.epsilon, method=method, omega = 1.9)
            diff.solve()
            plt.plot(diff.y, diff.c[:,2], label=f"{method.capitalize()}", color=colors[i], linestyle=linestyles[i])
        plt.title('Concentration at each y for different methods', fontsize=titlesize)
        plt.ylabel('y', fontsize=labelsize)
        plt.xlabel('Concentration', fontsize=labelsize)
        plt.legend(fontsize=ticksize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.tight_layout()
        plt.savefig('../figures/concentration_at_y.pdf')
        plt.show()

if __name__ == "__main__":
    # Example usage with stable parameters
    N = 50
    L = 1.0
    epsilon = 1e-6

    methods = [
        ('Jacobi', 1.0),
        ('Gauss-Seidel', 1.0),
        ('SOR', 1.7),
        ('SOR', 1.8),
        ('SOR', 1.9),
    ]

    diff = time_independent_diffusion(N=N, L=L, epsilon=epsilon, method='Jacobi')
    diff.plot_all_convergence()
    diff.plot_all_concentrations()

    # for method, omega in methods:
    #     print(f"\nSolving with {method.upper()} method:")
    #     diff = time_independent_diffusion(N=N, L=L, epsilon=epsilon,
    #                                       method=method, omega=omega)
    #     diff.solve()
    #     diff.plot()
    #     diff.plot_convergence()
