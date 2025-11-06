import numpy as np

class HistogramBinning:
    def __init__(self, n_bins=40, x_min=-1, x_max=1, style='lin', edges=None):
        self.n_bins = n_bins
        self.x_min = x_min
        self.x_max = x_max
        self.style = style

        if self.style == 'lin':
            self.edges = np.linspace(self.x_min, self.x_max, self.n_bins + 1)

        elif self.style == 'log':
            self.edges = np.logspace(self.x_min, self.x_max, self.n_bins + 1)

        elif self.style =='int':
            self.edges = np.linspace(-0.5, self.n_bins + 0.5, self.n_bins + 1)
            self.x_min = 0
            self.x_max = self.n_bins

        if edges:
            self.edges = edges[:]
            self.n_bins = len(edges) - 1
            self.x_min = edges[0]
            self.x_max = edges[-1]
            self.style = 'var'

    def find_bin(self, x):
        #add case for log
        if self.style == 'lin' or self.style == 'int':
            return int((x - self.x_min) / (self.x_max - self.x_min) * self.n_bins)     
        else:
            return np.digitize(x, self.edges)
            
class HistogramBase:
    def __init__(self, **kwargs):
        if 'bins' in kwargs.keys():
            self.bins = kwargs['bins']
        else:
            self.bins = HistogramBinning()

        self.edges = self.bins.edges
        self.n_bins = self.bins.n_bins
        self.n_entries = 0
        self.hist_w = np.zeros(self.n_bins, dtype=float)
        self.hist_n = []
        self.hist_sumw = []
        self.is_normalized = False

        """
        ## Transforms:
        Histograms with the same binning but instead of filling (x, w), now fill (x, f(x,w))
        Example: f(x,w) = w^2, to keep track of the weights squared per bin for error propagation
        """
        self.transforms = {}
        if 'transforms' in kwargs.keys():
            self.transforms = { k : (func, np.zeros(self.n_bins, dtype=float)) for k, func in kwargs['transforms'].items()}

        """
        ## Statistics
        Computes a statistic per batch fill and appends it to a list
        Example: f = mean, variance
        """
        self.stats = {}
        if 'stats' in kwargs.keys():
            self.stats = { k : (func, []) for k, func in kwargs['stats'].items()}

    def fill_from_array(self, x_arr, w_arr=None):

        #main histogram update
        tmp_hist_w, _ = np.histogram(x_arr, weights=w_arr, bins=self.edges)
        self.hist_w += tmp_hist_w

        #bookkeeping
        n_sample = len(x_arr)
        self.n_entries += n_sample
        self.hist_n.append(n_sample)
        if w_arr is None:
            self.hist_sumw.append(n_sample)
        else:
            self.hist_sumw.append(np.sum(w_arr))

        for (f, h) in self.transforms.values():
            tmp_hist_s, _ = np.histogram(x_arr, weights=w_arr, bins=self.edges)
            h += tmp_hist_s

        for (f, h) in self.stats.values():
            # print('Fill',f(x_arr,w_arr), my_variance(x_arr, w_arr), np.var(x_arr))
            h.append(f(x_arr, w_arr))
        self.is_normalized = False

    def normalize(self):

        if not self.is_normalized:
            sumw = np.sum(self.hist_w)
            self.hist_w = self.hist_w / sumw

            for _, h in self.transforms.values():
                h = h / sumw
            self.is_normalized = True


    def show(self):
        print(f"Histogram has {self.n_entries} entries, {len(self.hist_n)} batches")
        print("-"*60)
        print(f"{'index':>5}{'low':>12}{'high':>12}{'center':>12}{'value':>12}")
        for i in range(self.bins.n_bins):
            print(f"{i:5d}{self.edges[i]:12.3f}{self.edges[i+1]:12.3f}{0.5*(self.edges[i]+self.edges[i+1]):12.3f}{self.hist_w[i]:12.3f}")
        print("-"*60)
        print("Summary Statistics")
        for s, (_, q) in self.stats.items():
            print(f"{s} = {getattr(self,s)}")


## test 
if __name__ == "__main__":
    np.random.seed(42)
    def my_variance(x_arr, w_arr):
        return np.var(x_arr)
    

    n_add = 20
    n_samp = 200
    all_vals = np.random.normal(loc=0, scale=2, size=n_samp*n_add)
    h2 = HistogramBase(bins=HistogramBinning(20, -2, 2), stats={"var" : my_variance})
    print(h2.hist_sumw)
    h2.fill_from_array(all_vals)
    for k, (_,v) in h2.stats.items():
        print(k,v)
    print(np.var(all_vals))    

    print('--'*50)
    h = HistogramBase(bins=HistogramBinning(20, -2, 2), stats={"var" : my_variance})
    v = all_vals.reshape(n_add, n_samp)
    for r in v:
        h.fill_from_array(r)
    for k, (_,v) in h.stats.items():
        print(np.average(v,weights=h.hist_sumw))
        