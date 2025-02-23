import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# Plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

titlesize = 16
labelsize = 20
colors = sns.color_palette("Set2", 8)
ticksize = 18

class time_independent_diffusion:
    def __init__(self, N, L, epsilon=1e-5, max_iter=10000, method='Jacobi', omega=1.0):
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

        self.objects = np.zeros((N, N))

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

                # skip objects
                if self.objects[y, x] == 1:
                    continue

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

    def optimal_omega_binarysearch(self, tol=1e-3, max_iter=100, objects=False):
        """
        Finds the optimal omega for the SOR method using binary search.
        """
        omega_left, omega_right = 1.7, 2.0  # Optimal range for our diffusion problem
        best_omega = None
        best_iterations = np.inf

        for _ in range(max_iter):
            omega_mid = (omega_left + omega_right) / 2
            omega_test = omega_mid + tol

            diff_mid = time_independent_diffusion(N=self.N, L=self.L, epsilon=self.epsilon, method='SOR', omega=omega_mid)
            if objects:
                diff.add_rectangle(0.1, 0.9, 0.2, 0.5)
                diff.add_rectangle(0.7, 0.9, 0.8, 0.7)
                diff.add_polygon([(0.4, 0.3), (0.8, 0.5), (0.8, 0.3)])
            iterations_mid = diff_mid.solve()

            diff_test = time_independent_diffusion(N=self.N, L=self.L, epsilon=self.epsilon, method='SOR', omega=omega_test)
            if objects:
                diff.add_rectangle(0.1, 0.9, 0.2, 0.5)
                diff.add_rectangle(0.7, 0.9, 0.8, 0.7)
                diff.add_polygon([(0.4, 0.3), (0.8, 0.5), (0.8, 0.3)])
            iterations_test = diff_test.solve()

            if iterations_test < iterations_mid:
                omega_left = omega_mid
            else:
                omega_right = omega_mid

            if iterations_mid < best_iterations:
                best_omega = omega_mid
                best_iterations = iterations_mid

            if abs(omega_right - omega_left) < tol:
                break

        print(f"Optimal omega: {best_omega:.3f} (Converged in {best_iterations} iterations)")
        return best_omega

    def plot_omega_N(self, min_N=10, max_N=100, num_N=10, objects=False):
        """
        Plots the optimal omega as a function of N.
        """
        print(objects)
        N_values = np.linspace(min_N, max_N, num_N, dtype=int)
        optimal_omega_values = []
        for N in N_values:
            self.N = N
            optimal_omega_values.append(self.optimal_omega_binarysearch(objects=objects))
        # save in csv N_values and Omega values
        if objects:
            print("Saving results with objects")
            np.savetxt('../results/N_values_objects.csv', N_values, delimiter=',')
            np.savetxt('../results/optimal_omega_N_objects.csv', optimal_omega_values, delimiter=',')
        else:
            np.savetxt('../results/N_values.csv', N_values, delimiter=',')
            np.savetxt('../results/optimal_omega_N.csv', optimal_omega_values, delimiter=',')
        N_values = np.loadtxt('../results/N_values.csv', delimiter=',')
        optimal_omega_values = np.loadtxt('../results/optimal_omega_N.csv', delimiter=',')
        N_values_objects = np.loadtxt('../results/N_values_objects.csv', delimiter=',')
        optimal_omega_values_objects = np.loadtxt('../results/optimal_omega_N_objects.csv', delimiter=',')
        plt.figure(figsize=(8, 5))
        plt.plot(N_values, optimal_omega_values, 'o-', color=colors[1], label = 'No objects')
        #plt.plot(N_values_objects, optimal_omega_values_objects, 'o-', color=colors[2], label = 'With objects')
        plt.xlabel(r'Intervals ($N$)', fontsize=labelsize+4)
        plt.ylabel(r'Optimal $\omega$', fontsize=labelsize+4)
        plt.xticks(fontsize=ticksize+4)
        plt.yticks(fontsize=ticksize+4)
        plt.legend(fontsize=ticksize+4)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('../figures/optimal_omega_N.pdf')
        plt.show()

    def plot(self):
        """Plot the current state of the system as a 2D color map"""
        plt.figure(figsize=(8, 8))

        im = plt.imshow(self.c,
                       extent=[0, self.L, 0, self.L],
                       origin='lower',
                       cmap='viridis',
                       aspect='equal',
                       vmin=0, vmax=1)

        # Overlay objects
        if self.objects.any():
            object_color = 'gray'
            object_array = np.ma.masked_where(self.objects == 0, self.objects)
            plt.imshow(object_array,
                    extent=[0, self.L, 0, self.L],
                    origin='lower',
                    cmap=object_color,
                    alpha=0.5)
        plus = 4
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Concentration', fontsize=labelsize+plus)
        cbar.ax.tick_params(labelsize=ticksize+plus)
        plt.xlabel('x', fontsize=labelsize+ plus)
        plt.ylabel('y', fontsize=labelsize+ plus)
        #plt.title(f'Steady State Concentration Distribution ({self.method})', fontsize=titlesize)
        plt.yticks(fontsize=ticksize+ plus)
        plt.xticks(fontsize=ticksize+ plus)
        plt.tight_layout()
        plt.savefig('../figures/concentration_distribution.pdf')
        plt.show()

    def plot_convergence(self):
        """
        Plot the convergence history (delta vs iterations).
        """
        plt.figure(figsize=(8, 8))
        plt.semilogy(self.delta_history)
        plt.title('Convergence of Jacobi Iteration', fontsize=titlesize)
        plt.xlabel(r'Iterations ($k$)', fontsize=labelsize)
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
        omega_values = [1.0, 1.0, 1.7, 1.8, 1.92] # add more indicative values?

        for i, method in enumerate(['Jacobi', 'Gauss-Seidel','SOR', 'SOR', 'SOR']):
            omega_temp = omega_values[i]
            diff = time_independent_diffusion(N=self.N, L=self.L, epsilon=self.epsilon,
                                              method=method, omega=omega_temp)
            diff.solve()
            plt.semilogy(diff.delta_history, label=rf"{method} ($\omega$ = {omega_temp})", color=colors[i])
        #plt.title(f'Convergence of different Methods', fontsize=titlesize)
        plt.xlabel(r'Iterations ($k$)', fontsize=labelsize)
        plt.ylabel(r'Maximum Change $\delta$ (log scale)', fontsize=labelsize)
        plt.grid(True)
        plt.legend(fontsize=ticksize)
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.tight_layout()
        plt.savefig('../figures/convergence.pdf')
        plt.show()

    def test_2D_simulation(self):
        """""Test the correctness of the simulation by comparing the final state to the analytical solution"""
        diff = time_independent_diffusion(N=self.N, L=self.L, epsilon=self.epsilon, method=self.method, omega=self.omega)
        diff.solve()
        c_analytical = diff.y
        for x in range(self.N):
            cx = diff.c[:,x]
            np.testing.assert_allclose(cx, c_analytical, rtol=1e-2)
        print

    def add_line(self, x0, y0, x1, y1):
        """
        Add a line as a sink to the system domain. This value will be 0.

        Args:
            x0, y0: Starting point of the line ratios between 0 and 1
            x1, y1: Ending point of the line ratios between 0 and 1

        based off the simple DDA line generation algorithm:
        https://www.geeksforgeeks.org/dda-line-generation-algorithm-computer-graphics/
        """
        assert all(0 <= el <= 1 for el in [x0, y0, x1, y1]), \
               "line points must be in the range [0, 1]"
        coords = set()

        # map xs to the system domain
        x0 = int(x0 * self.N)
        x1 = int(x1 * self.N)

        # exclude the upper and lower bounds in the y-mapping
        y0 = 1 + int(y0 * self.N - 2)
        y1 = 1 + int(y1 * self.N - 2)

        dx = x1 - x0
        dy = y1 - y0

        steps = max(abs(dx), abs(dy))

        xinc = dx/steps
        yinc = dy/steps

        # start with 1st point
        x = x0
        y = y0
        coords.add((x0, y0))

        for i in range(steps):
            x = x + xinc
            y = y + yinc
            coords.add((round(x), round(y)))

        # add the objects to the object grid
        for x, y in coords:
            self.objects[y, x] = 1

    def add_polygon(self, points, method='line'):
        """
        Add a polygon as a sink to the system domain. This value will be 0.

        Args:
            points: List of points (x, y) that define the polygon
            method: Method to connect the points. 'line' connects each point to the next.
        """
        assert len(points) >= 3, "A polygon must have at least 3 points"
        assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in points), \
               "Polygon points must be in the range [0, 1]"
        if method == 'line':
            for i in range(len(points)):
                x0, y0 = points[i]
                x1, y1 = points[(i+1) % len(points)]
                self.add_line(x0, y0, x1, y1)
        else:
            raise ValueError(f"Unknown method: {method}")

    def add_rectangle(self, x0, y0, x1, y1):
        """
        Add a rectangle as a sink to the system domain. This value will be 0.

        Args:
            x0, y0: Bottom left corner of the rectangle ratios between 0 and 1
            x1, y1: Top right corner of the rectangle ratios between 0 and 1
        """
        assert all(0 <= el <= 1 for el in [x0, y0, x1, y1]), \
               "rectangle corners must be in the range [0, 1]"
        points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        self.add_polygon(points)

    def add_square(self, x0, y0, size):
        """
        Add a square as a sink to the system domain. This value will be 0.

        Args:
            x0, y0: Bottom left corner of the square ratios between 0 and 1
            size: Size of the square
        """
        assert all(0 <= el <= 1 for el in [x0, y0]), \
               "square corner must be in the range [0, 1]"
        assert 0 <= size <= 1, "size must be in the range [0, 1]"
        x1 = x0 + size
        y1 = y0 + size
        self.add_rectangle(x0, y0, x1, y1)

    def add_circle(self, x0, y0, r):
        """
        Add a circle as a sink to the system domain. This value will be 0.

        Args:
            x0, y0: Center of the circle ratios between 0 and 1
            r: Radius of the circle
        """
        assert all(0 <= el <= 1 for el in [x0, y0]), \
               "circle center must be in the range [0, 1]"
        x0 = int(x0 * self.N)
        y0 = 1 + int(y0 * self.N - 2)
        r = int(r * self.N)

        for x in range(self.N):
            for y in range(1, self.N - 1):
                if (x - x0) ** 2 + (y - y0) ** 2 <= r ** 2:
                    self.objects[y, x] = 1

    def add_semi_circle(self, x0, y0, r):
        """
        Add a top-half semicircle as a sink to the system domain.

        Args:
            x0, y0: Center of the semicircle (normalized between 0 and 1).
            r: Radius of the semicircle (normalized between 0 and 1).
        """
        assert all(0 <= el <= 1 for el in [x0, y0]), \
            "Semicircle center must be in the range [0, 1]"

        # Convert normalized coordinates to grid indices
        x0 = int(x0 * self.N)
        y0 = 1 + int(y0 * self.N - 2)
        r = int(r * self.N)

        # Iterate through the grid and mark only the top-half of the circle
        for x in range(self.N):
            for y in range(1, self.N - 1):
                if (x - x0) ** 2 + (y - y0) ** 2 <= r ** 2 and y >= y0:
                    self.objects[y, x] = 1  # Fill the semicircle


    def init_objects(self):
        """
        Initialize some random objects in the system domain.
        """
        # horizontal line
        self.add_line(0.075, 0.9, 0.125, 0.9)
        # vertical line
        self.add_line(0.25, 0.925, 0.25, 0.875)
        # lil triangle (polygon)
        self.add_polygon([(0.375, 0.85), (0.4, 0.95), (0.44, 0.85)])
        # rectangle
        self.add_rectangle(0.55, 0.925, 0.6, 0.85)
        # # circle
        self.add_circle(0.85, 0.9, 0.075)

    def fill_triangle(self, p1, p2, p3):
        """
        Fill a triangle in the system domain.

        Args:
            p1, p2, p3: Tuples (x, y) representing the triangle vertices (normalized between 0 and 1).
        """
        assert all(0 <= x <= 1 and 0 <= y <= 1 for x, y in [p1, p2, p3]), \
            "Triangle points must be in the range [0, 1]"

        # Convert normalized coordinates to grid space
        x1, y1 = int(p1[0] * self.N), int(p1[1] * self.N)
        x2, y2 = int(p2[0] * self.N), int(p2[1] * self.N)
        x3, y3 = int(p3[0] * self.N), int(p3[1] * self.N)

        # Compute bounding box
        min_x = max(min(x1, x2, x3), 0)
        max_x = min(max(x1, x2, x3), self.N - 1)
        min_y = max(min(y1, y2, y3), 0)
        max_y = min(max(y1, y2, y3), self.N - 1)

        # Barycentric coordinate method to determine if a point is inside the triangle
        def is_inside(px, py):
            detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            alpha = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / detT
            beta = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / detT
            gamma = 1 - alpha - beta
            return 0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1

        # Iterate through the bounding box and fill in the triangle
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if is_inside(x, y):
                    self.objects[y, x] = 1  # Fill the triangle

    def init_heart(self):
        self.fill_triangle((0.5, 0.3), (0.3, 0.6), (0.7, 0.6))  # Fills a triangle
        self.add_semi_circle(0.4, 0.6, 0.1)
        self.add_semi_circle(0.6, 0.6, 0.1)


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
    print("Testing against analytical solution...")
    try:
        diff.test_2D_simulation()
        print("Test passed! Numerical solution matches analytical solution within tolerance.")
    except AssertionError as e:
        print("Test failed:", e)

    # diff.plot_all_convergence()
    # diff.plot_omega_N()
    #diff.optimal_omega_binarysearch()
    #diff.plot_all_concentrations()

    # for method, omega in methods:
    #     print(f"\nSolving with {method.upper()} method:")
    #     diff = time_independent_diffusion(N=N, L=L, epsilon=epsilon,
    #                                       method=method, omega=omega)
    #     diff.solve()
    #     diff.plot()
    #     diff.plot_convergence()

    def test_objects():
        N = 200
        L = 1.0
        epsilon = 1e-6

        methods = [
            ('SOR', 1.92),
        ]

        for method, omega in methods:
            print(f"\nSolving with {method.upper()} method:")
            diff = time_independent_diffusion(N=N, L=L, epsilon=epsilon,
                                            method=method, omega=omega)
            diff.init_heart()

            diff.solve()
            diff.plot()

    #test_objects()

    def test_omega_objects():
        N = 50
        L = 1.0
        epsilon = 1e-6

        diff = time_independent_diffusion(N=N, L=L, epsilon=epsilon, method='SOR')
        diff.init_heart()
        # set to true so then adds the heart to the diffusion objects
        #diff.optimal_omega_binarysearch(objects=True)
        diff.plot_omega_N(objects=True)

    test_omega_objects()
