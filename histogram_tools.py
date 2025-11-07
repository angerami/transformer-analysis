import numpy as np
import matplotlib.pyplot as plt
import pickle

class HistogramBase:

    global_opts = { "batch_mode" : False}
    metadata_attributes = ['name', 'n_fill', 'n_entries', 'stats_values', 'sum_w', 'sum_w2']

    def __init__(self, name='h', use_weights=False, used_weights_sq=False, **kwargs):

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

        #statistics
        self.stats_functions = {}
        self.stats_values = {}

        if 'stats' in kwargs.keys():
            self.set_stats(kwargs['stats'])

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

    #helper to set or reset stats
    def set_stats(self, stats_config):
        self.stats_functions = { k : v for k, v in stats_config.items()}

    
    ############################################################
    ### Accessors
    ############################################################

    def hist(self):
        return self.histograms['h']
    
    def stats(self):
        return self.stats_values
    
    def get_statistic(self, name):
        return self.stats_values[name]
    
    def hist_norm(self, var_bin=False, eps=1e-10):
        p = self.hist()[:] / self.n_entries
        p = p / np.diff(self.bins) if var_bin else p / self.bin_dx
        return p + eps

    ############################################################
    ### Filling functions
    ############################################################

    def fill_stats(self, x_arr):
        self.stats_values = { k : f(x_arr) for k, f in self.stats_functions.items()}

    def fill(self, x_arr, w_arr=None):

        #bookkeeping
        self.n_fill += 1
        n_sample = len(x_arr)
        self.n_entries += n_sample
        self.hist_n.append(n_sample)
        
        #main histogram update
        tmp_hist,_ = np.histogram(x_arr, bins=self.bins)
        self.histograms['h'] += tmp_hist

        self.fill_stats(x_arr)

        #weight updates
        if self.use_weights:
            if not w_arr or len(w_arr) != len(x_arr): 
                tmp_hist, _ = np.histogram(x_arr, weights=None, bins=self.bins)
            else:
                tmp_hist, _ = np.histogram(x_arr, weights=w_arr, bins=self.bins)
            self.histograms['h_w'] += tmp_hist
            self.hist_sumw.append(np.sum(tmp_hist))

        if self.use_weights_sq:
            if not w_arr or len(w_arr) != len(x_arr): 
                tmp_hist, _ = np.histogram(x_arr, weights=None, bins=self.bins)
            else:
                tmp_hist, _ = np.histogram(x_arr, weights=w_arr*w_arr, bins=self.bins)
            self.histograms['h_w2'] += tmp_hist
            self.hist_sumw2.append(np.sum(tmp_hist))

    ############################################################
    ### I/O: persistification and helpers
    ############################################################

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
            self.bin_dx = self.bins[1] - self.bins[0]

    def save(self, filename='hists.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.to_dict())
    
    def load(self, filename='hists.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.from_dict(data)

    ############################################################
    ### Data Inspection
    ############################################################

    def get_bin_centers(self):        
        if self.bin_centers is None:
            self.bin_centers = 0.5 * (self.bins[:-1] + self.bins[1:])
        return self.bin_centers
    
    def show(self, verbosity='summary'):
        print(f"Histogram has {self.n_entries} entries, {len(self.hist_n)} batches")
        print("Statistics")
        for k, v in self.stats_values.items():
            print(f"{k} = {v:.3f}")
        if verbosity == 'bins':
            print("-"*60)
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

############################################################
### HistogramGroup class
### container for HistogramBase objects with identical attributes
### arranged in 2D array
############################################################
class HistogramGroup():
    def __init__(self, bins=None, prefix='h', n_layers=4, n_heads=8, **kwargs):
        self.prefix = prefix
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.bins = bins
        self.histogroup = {}
        self.metadata = {}

        if bins is not None:
            for layer_idx in range(self.n_layers):
                for head_idx in range(self.n_heads):
                    self.histogroup[(layer_idx, head_idx)] =  HistogramBase(bins=bins, name=f'h_L{layer_idx:03d}_H{head_idx:03d}',**kwargs)

    def __getitem__(self, key):
        return self.histogroup[key]
    
    def __setitem__(self, key, value):
        self.histogroup[key] = value

    def __iter__(self):
        return iter(self.histogroup.items())

    def fill(self, layer_idx, head_idx, x_arr, w_arr=None):
        self.histogroup[(layer_idx, head_idx)].fill(x_arr,w_arr)
    
    def metadata_to_dict(self):
        return { attr_name : getattr(self,attr_name) for attr_name in ['n_layers', 'n_heads', 'prefix']}
    
    def set_metadata_from_dict(self, meta_dict):
        for k, v in meta_dict.items():
            setattr(self, k, v)

    def save(self, filename='h.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({'metadata' : self.metadata_to_dict(), 'histogroup' : {k : v.to_dict() for k,v in self}, 'bins' : self.bins}, f)
    
    def load(self, filename='h.pkl'):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.set_metadata_from_dict(data['metadata'])
            self.bins = data['bins']
            for k,v in data['histogroup'].items():
                self.histogroup[k] = HistogramBase(bins=self.bins)
                self.histogroup[k].from_dict(v)

    def extract_histos(self):
        return {k : v.hist() for k, v in self}

    def extract_stats(self, stat_name):
        return {k : v.get_statistic(stat_name) for k, v in self}

    def analyze_histos(self, ana_func):
        return {k : ana_func(v) for k, v in self}


if __name__ == '__main__':

    def MY_TEST_MESSAGE(test_name, result, condition):

        condition = 'PASSED' if condition else 'FAILED'
        print('-'*40 + f'\n{test_name} Test: {result} \n{condition}\n')
    
    def HIST_DIFF(h1, h2):
        return np.sum(np.abs(h1 - h2))

    test_fill = False
    test_IO = False
    test_sums = False
    test_stats = False

    # test_fill = True
    # test_IO = True
    # test_sums = True
    # test_stats = True

    inspect = True

    np.random.seed(123)
    test_fname = 'test.pkl'
    err_tol = 1e-9
    n_layers, n_heads, n_samp = 2, 3, 1000
    bins = np.linspace(-2, 2, 20)
    data = np.random.normal(size=n_layers*n_heads*n_samp).reshape((n_layers, n_heads,n_samp))
    #test fill
    if test_fill:
        h1 = HistogramBase(bins=bins)
        h1.fill(np.random.uniform(bins[0], bins[-1], size=n_samp))
        err = np.abs(np.sum(h1.hist())) - n_samp
        MY_TEST_MESSAGE("FILL", err, err < err_tol)

    # test order of sums (associativity)
    if test_sums:
        #h1 : single fill
        h1 = HistogramBase(bins=bins)
        h1.fill(data.ravel())

        #h2 : many fills
        h2 = HistogramBase(bins=bins)
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                h2.fill(data[layer_idx, head_idx])
        #compare outputs
        err = HIST_DIFF(h1.hist(),h2.hist())
        MY_TEST_MESSAGE('SUMS', err, err < err_tol)

    if test_stats:
        stats_config = {'mean' : np.mean, 'std' : np.std, 'max' : np.max}
        h1 = HistogramBase(bins=bins, stats=stats_config)
        flat = data.ravel()
        h1.fill(flat)

        for k, v in stats_config.items():
            err += np.abs(h1.get_statistic(k) - v(flat))
        MY_TEST_MESSAGE('Stats', err, err < err_tol,)

    #test I/0
    if test_IO:
        stats_config = {'mean' : np.mean, 'std' : np.std, 'max' : np.max}
        #create and fill
        ha_result = {k : [] for k in stats_config.keys()}

        ha = HistogramGroup(bins=bins, n_layers=n_layers, n_heads=n_heads, stats=stats_config)
        for (layer_idx, head_idx), h in ha.histogroup.items():
           h.fill(data[layer_idx,head_idx])
           for k in stats_config.keys():
               ha_result[k].append(h.get_statistic(k))
        ha.save(test_fname)

        #read back from disk
        hb = HistogramGroup()
        hb.load(test_fname)

        #compare outputs
        err = 0
        stat_err = 0
        for k, v in hb.histogroup.items():
            err += HIST_DIFF(v.hist(),ha[k].hist())
            for s in stats_config.keys():
               stat_err += np.abs(v.get_statistic(s) - ha[k].get_statistic(s))
        MY_TEST_MESSAGE('I/O', err, err < err_tol)
        if test_stats:
            MY_TEST_MESSAGE('I/O stat', stat_err, stat_err < err_tol)

    if inspect:
        stats_config = {'mean' : np.mean}
        h1 = HistogramBase(bins=bins, stats=stats_config)
        flat = data.ravel()
        h1.fill(flat)
        h1.show()



