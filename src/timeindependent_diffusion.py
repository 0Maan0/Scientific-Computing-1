import numpy as np  
import matplotlib.pyplot as plt

class time_independent_diffusion:
    def __init__(self, N, L, epsilon = 1e-5):
        self.N = N
        self.L = L
        self.dx = L / N

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
        
            for x in range(1, self.N):
                for y in range(1, N-1):
                    c_next[x, y] = 0.25 * (self.c[x+1, y] + c[x-1, y] + c[x, y+1] + c[x, y-1])
