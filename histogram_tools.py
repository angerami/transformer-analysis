import numpy as np
import matplotlib.pyplot as plt
import pickle

class HistogramBase:

    global_opts = { "batch_mode" : False}
    metadata_attributes = ['name', 'n_fill', 'n_entries', 'sum_w', 'sum_w2']

    def __init__(self, name='h', use_weights=True, used_weights_sq=True, **kwargs):

        #Binning
        if 'bins' in kwargs.keys():
            self.bins = kwargs['bins']
        else:
            self.bins = np.linspace(-1, 1, 5)
        self.bin_dx = self.bins[1] - self.bins[0]
        self.name = name
        self.use_weights = use_weights
        self.use_weights_sq = used_weights_sq
        self.n_fill = 0
        self.histograms = {'bins' : self.bins}

        n_bins = len(self.bins) - 1
        #core histograms
        self.n_entries = 0
        self.histograms['h'] = np.zeros(n_bins, dtype=float)
        self.hist_n = []

        #weighted
        self.sum_w = 0
        if self.use_weights:
           self.histograms['h_w'] = np.zeros(n_bins, dtype=float)
           self.hist_sumw = []

        self.sum_w2 = 0
        if self.use_weights_sq:
            self.histograms['h_w2'] = np.zeros(n_bins, dtype=float)
            self.hist_sumw2 = []

        self.bin_centers = None

    def get_bin_centers(self):        
        if self.bin_centers is None:
            self.bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        return self.bin_centers


    def fill(self, x_arr, w_arr=None):

        #bookkeeping
        self.n_fill += 1
        n_sample = len(x_arr)
        self.n_entries += n_sample
        self.hist_n.append(n_sample)
        
        #main histogram update
        tmp_hist,_ = np.histogram(x_arr, bins=self.bins)
        self.histograms['h'] += tmp_hist

        #weight updates
        if not w_arr or len(w_arr) != len(x_arr): 
            w_arr = np.ones_like(x_arr)

        if self.use_weights:
            tmp_hist, _ = np.histogram(x_arr, weights=w_arr, bins=self.bins)
            self.histograms['h_w'] += tmp_hist
            self.hist_sumw.append(np.sum(tmp_hist))

        if self.use_weights_sq:
            tmp_hist, _ = np.histogram(x_arr, weights=w_arr**2, bins=self.bins)
            self.histograms['h_w2'] += tmp_hist
            self.hist_sumw2.append(np.sum(tmp_hist))

    #persistification and helpers
    def metadata_to_dict(self):
        return { attr_name : getattr(self,attr_name) for attr_name in HistogramBase.metadata_attributes}
    
    def set_metadata_from_dict(self, meta_dict):
        for k, v in meta_dict.items():
            setattr(self, k, v)
    
    def to_dict(self, include_bins=False):
        out_dict = {'metadata' : self.metadata_to_dict(), 'histograms': self.histograms}
        if include_bins:
            out_dict['bins'] = self.bins
        return out_dict
    
    def from_dict(self, input):
        self.set_metadata_from_dict(input['metadata'])
        self.histograms = input['histograms']
        if 'bins' in input.keys():
            self.bins = input['bins']
        
    def save(self, filename='hists.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.to_dict())
    
    def load(self, filename='hists.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.from_dict(data)

    #inspection
    def show(self, verbosity='summary'):
        print(f"Histogram has {self.n_entries} entries, {len(self.hist_n)} batches")
        print("-"*60)
        if verbosity == 'bins':
            print(f"{'Index':>5}{'Low':>12}{'High':>12}{'Center':>12}{'Value':>12}")
            for i in range(len(self.bins) - 1):
                print(f"{i:5d}{self.bins[i]:12.3f}{self.bins[i+1]:12.3f}{0.5*(self.bins[i]+self.bins[i+1]):12.3f}{self.histograms['h'][i]:15.0f}")
            print("-"*60)

    def draw(self, name='h', same=False, log=False, normalized=False, out=None, **kwargs):
        h = self.histograms[name].astype(float)
        if normalized:
            h = h / h.sum()                
        plt.figure()
        plt.bar(self.get_bin_centers(),h, width=self.bin_dx, alpha=0.2, label=self.name, **kwargs)
        if log:
            plt.yscale('log')
        plt.title(name)
        plt.show(block=False)
        if out is not None:
            plt.savefig(f"{name}.{out}")

class HistogramGroup():
    def __init__(self, bins=None, prefix='h', n_layers=4, n_heads=8):
        self.prefix = prefix
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.bins = bins
        self.histogroup= {}
        self.metadata = {}

        if bins is not None:
            for layer_idx in range(self.n_layers):
                for head_idx in range(self.n_heads):
                    self.histogroup[(layer_idx, head_idx)] =  HistogramBase(bins=bins, name='h_L{layer_idx:03d}_H{head_idx:03d}')

    def fill(self, layer_idx, head_idx, x_arr, w_arr=None):
        self.histogroup[(layer_idx, head_idx)].fill(x_arr,w_arr)
    
    def metadata_to_dict(self):
        return { attr_name : getattr(self,attr_name) for attr_name in ['n_layers', 'n_heads', 'prefix']}
    
    def set_metadata_from_dict(self, meta_dict):
        for k, v in meta_dict.items():
            setattr(self, k, v)

    def save(self, filename='h.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({'metadata' : self.metadata_to_dict(), 'histogroup' : {k : v.to_dict() for k, v in self.histogroup.items()}, 'bins' : self.bins}, f)
    
    def load(self, filename='h.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.set_metadata_from_dict(data['metadata'])
            self.bins = data['bins']
            for k,v in data['histogroup'].items():
                self.histogroup[k] = HistogramBase(bins=self.bins)
                self.histogroup[k].from_dict(v)
        
def load_group_from_file(filename):
    g = HistogramGroup()
    g.load(filename)
    return g



if __name__ == '__main__':
    bins = np.linspace(-2.5, 2.5, 11)
    hg = HistogramGroup(bins=bins, n_layers=2, n_heads=3)
    hg.save("test.pkl")
    ha = HistogramGroup()
    ha.load("test.pkl")
    for i in range(ha.n_layers):
        for j in range(ha.n_heads):
            v = np.random.normal(size=1000)
            ha.histogroup[(i, j)].fill(v)
        ha.save(filename="t1.pkl")
    hb = load_group_from_file("t1.pkl")

    for k, v in hb.histogroup.items():
        print(k)
        print(v.histograms)