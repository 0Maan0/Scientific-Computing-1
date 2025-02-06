import numpy as np


class Wave:
    def __init__(self, N, L, v=3*10**8):
        self.N = N
        self.L = L
        self.v = v
        self.dx = L / N

        # discretized points
        self.x = np.linspace(0, L, N)

        self.t = 0
        self.y = np.zeros(N)

        # boundary conditions 0 on start and end
        self.y[0] = 0
        self.y[-1] = 0

    def step(self):
        pass
