import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
from scipy.special import erfc
import seaborn as sns

# Plotting parameters
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

titlesize = 16
labelsize = 20
colors = sns.color_palette("Set2", 8)
ticksize = 18

# 1.2 The Time Dependent Diffusion Equation
class Diffusion:
    def __init__(self, N, L, dt=0.001, D=1, tol=1e-3):
        self.N = N
        self.L = L
        self.dx = L / N
        self.dt = dt
        self.D = D
        self.delta = 1  # high starting value
        self.tol = tol

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

    def test_2D_simulation(self, n_terms=10):
        """""Test the correctness of the 2D simulation by comparing the final state to the analytical solution"""
        c_analytical = np.zeros((self.N, self.N))

        for j in range(self.N):
            y = self.y[j]
            sum_terms = 0
            for i in range(n_terms):
                term1 = erfc((1 - y + 2 * i) / (2 * np.sqrt(self.D * self.t)))
                term2 = erfc((1 + y + 2 * i) / (2 * np.sqrt(self.D * self.t)))
                sum_terms += (term1 - term2)

            # apply this value to all x positions at this y level (i.e. row)
            c_analytical[j, :] = sum_terms

        # for very large t, verify we approach the linear solution c(y) = y
        if self.t > 1.0:
            expected_linear = self.y.reshape(-1, 1)
            np.testing.assert_allclose(c_analytical, expected_linear, rtol=1, atol=1e-6)

        # compare numerical to analytical solution
        np.testing.assert_allclose(self.c, c_analytical,rtol=1, atol=1e-6)
        
    def plot_analytical(self, times=[0.001, 0.01, 0.1, 1.0], n_terms=10):
        """
        Plot analytical solution for different t.
        """
        plt.figure(figsize=(10, 6))
        
        # Line styles for different times
        linestyles = ['-', '--', ':', '-.']
        
        # Fine grid for smooth curves
        y_fine = np.linspace(0, self.L, 200)
        
        for i, t in enumerate(times):
            c_analytical = np.zeros_like(y_fine)
            
            # Calculate analytical solution
            for j, y in enumerate(y_fine):
                sum_terms = 0
                for k in range(n_terms):
                    term1 = erfc((1 - y + 2 * k) / (2 * np.sqrt(self.D * t)))
                    term2 = erfc((1 + y + 2 * k) / (2 * np.sqrt(self.D * t)))
                    sum_terms += (term1 - term2)
                c_analytical[j] = sum_terms
            
            plt.plot(y_fine, c_analytical,
                    linestyle=linestyles[i],
                    label=f'Dt={t:.3f}')
            
        plt.xlabel('c(y)', fontsize=labelsize+4)
        plt.ylabel('Concentration(c)', fontsize=labelsize+4)
        plt.xticks(fontsize=ticksize+4)
        plt.yticks(fontsize=ticksize+4)
        plt.grid(True)
        plt.legend(fontsize=ticksize+4)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('./figures/analytical_plot.pdf')
        plt.show()

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
        self.delta = np.max(np.abs(c_next - self.c))
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
        cbar = plt.colorbar(im, label='Concentration (c)')
        cbar.ax.tick_params(labelsize=ticksize+8)
        cbar.set_label('Concentration (c)', fontsize=labelsize+8)
        plt.xlabel('x', fontsize=labelsize+8)
        plt.ylabel('y', fontsize=labelsize+8)
        plt.xticks(fontsize=ticksize+8)
        plt.yticks(fontsize=ticksize+8)
        plt.title(f't = {self.t:.3f}', fontsize=labelsize+4)
        plt.savefig(f'./figures/diffusion_t_{self.t:.3f}.pdf', bbox_inches='tight')
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
        # size of text colorbar
        plt.colorbar(im, label='Concentration')
        ax.set_xlabel('x', fontsize=labelsize)
        ax.set_ylabel('y', fontsize=labelsize)

        def update(frame):
            # Do multiple steps per frame
            for _ in range(steps_per_frame):
                self.step()

                # stopping condition
                if self.delta < self.tol:
                    anim.event_source.stop()
                    print(f"Stopping early at frame {frame}, t = {self.t:.3f}, delta = {self.delta:.6f}")
                    break

            im.set_array(self.c)
            ax.set_title(f't = {self.t:.3f}, frame = {frame * steps_per_frame}')
            return [im]

        anim = FuncAnimation(fig, update, frames=num_frames,
                        interval=interval, blit=False)
        # anim.save(filename="../figures/timedep_diffusion.mkv", writer="ffmpeg")
        plt.show()
        return anim


if __name__ == "__main__":
    # Example usage with stable parameters
    N = 50
    L = 1.0
    D = 1.0
    dx = L/N
    dt = 0.0001

    diff = Diffusion(N=N, L=L, dt=dt, D=D, tol=1e-6)
    diff.animate(num_frames=1000, interval=1, steps_per_frame=10)
    
    def plot_times():
        fp_tol = 1e-10
        diff = Diffusion(N=N, L=L, dt=dt, D=D, tol=1e-6)
        # plot 0
        diff.plot()
        # plot rest of the times
        times = [0.001, 0.01, 0.1, 1.0]
        while True:
            diff.step()
            for t in times:
                if np.isclose(diff.t, t, atol=fp_tol):
                    diff.plot()    
    plot_times()

    def test():
        print("Testing against analytical solution...")
        try:
            diff.test_2D_simulation(n_terms=2)
            print("Test passed! Numerical solution matches analytical solution within tolerance.")
        except AssertionError as e:
            print("Test failed:", e)

    # test()
    
    diff.plot_analytical()


