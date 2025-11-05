import numpy as np

class HistoBinning:
    def __init__(self, n_bins=40, x_min=-1, x_max=1, style='lin', edges=None):
        self.n_bins = n_bins
        self.x_min = x_min
        self.x_max = x_max
        self.style = style

        if self.style == 'lin':
            self.edges = np.linspace(self.n_bins, self.x_min, self.x_max)

        elif self.style == 'log':
            self.edges = np.logspace(self.n_bins, self.x_min, self.x_max)

        elif self.style =='int':
            self.edges = np.linspace(self.n_bins, -0.5, self.n_bins + 0.5)
            self.x_min = 0
            self.x_max = self.n_bins

        if edges:
            self.edges = edges[:]
            self.n_bins = len(edges) - 1
            self.x_min = edges[0]
            self.x_max = edges[-1]
            self.style = 'var'

    def find_bin(self, x):
        if self.style == 'var':
            return np.digitize(x, self.edges)
        return int((x - self.x_min) / (self.x_max - self.x_min) * self.n_bins)     
