import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from histogram_utils import config_standard, config_default

class HistogramBase:

    metadata_attributes = ['name', 'n_fill', 'n_entries']

    def __init__(self, weight_type='W_Q',  bins_dict=None, stats={},  do_svd=False, use_density=True):

        if not bins_dict:
            self.w_bins = config_default['w_bins']
            self.sv_bins = config_default['sv_bins']
        else:
            self.w_bins = bins_dict['w_bins']
            self.sv_bins = bins_dict['sv_bins']


        self.weight_type = weight_type
        self.n_fill = 0
        self.n_entries = 0
        
        n_bins = len(self.w_bins) - 1
        self.histograms = {'P_W' :  np.zeros(n_bins, dtype=float)}
        
        #statistics
        self.set_stats(stats)
        self.stats_values = {k : np.nan for k in stats.keys()}

        self.do_svd = do_svd
        self.use_density = use_density
    
    ############################################################
    ### Configuration
    ############################################################
    
    def set_stats(self, stats_config):
        self.stats_functions = { k : v for k, v in stats_config.items()}

    ############################################################
    ### Accessors
    ############################################################

    def hist(self, hname='P_W'):
        return self.histograms[hname]
    
    def stats(self):
        return self.stats_values
    
    def get_statistic(self, name):
        return self.stats_values[name]
    
    def get_bin_centers(self):
        return (self.w_bins[:-1] + self.w_bins[1:]) / 2
    
    ############################################################
    ### Filling functions
    ############################################################

    def fill_stats(self, x_arr):
        self.stats_values = { k : f(x_arr) for k, f in self.stats_functions.items()}

    def fill(self, W_tensor):
        
        #main histogram update
        x_arr = W_tensor.flatten().detach().cpu().numpy()
        tmp_hist,_ = np.histogram(x_arr, bins=self.w_bins, density=self.use_density)
        self.histograms['P_W'] += tmp_hist
        self.fill_stats(x_arr)

        #bookkeeping
        self.n_fill += 1
        self.n_entries += len(x_arr)
        del x_arr
        if self.do_svd:
            _, S, _ = torch.linalg.svd(W_tensor)
            self.histograms['SVD'] = S.detach().cpu().numpy()
            del S
            P_l,_ =  np.histogram(self.histograms['SVD'], bins=self.sv_bins)
            self.histograms['P_l'] = P_l



        
############################################################
### HistogramGroup class
### container for HistogramBase objects with identical attributes
### arranged in 2D array
############################################################
class HistogramGroup():
    def __init__(self, weight_type='W_Q', n_layers=4, n_heads=8, bins_dict=None, stats={}, do_svd=False, use_density=True, **kwargs):
        self.weight_type = weight_type
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.histogroup = {}
        self.metadata = {}
        self.stats = stats
        if not bins_dict:
            self.w_bins = config_default['w_bins']
            self.sv_bins = config_default['sv_bins']
        else:
            self.w_bins = bins_dict['w_bins']
            self.sv_bins = bins_dict['sv_bins']

        for layer_idx in range(self.n_layers):
            for head_idx in range(self.n_heads):
                self.histogroup[(layer_idx, head_idx)] =  HistogramBase(weight_type=weight_type,
                                                                        bins_dict=bins_dict,
                                                                        stats = stats,
                                                                        do_svd = do_svd,
                                                                        use_density=use_density)

    def __getitem__(self, key):
        return self.histogroup[key]
    
    def __setitem__(self, key, value):
        self.histogroup[key] = value

    def __iter__(self):
        return iter(self.histogroup.items())
    
    @classmethod
    def standard(cls,  weight_type, n_layers, n_heads):
        return cls(
            weight_type=weight_type, n_layers=n_layers, n_heads=n_heads, 
            do_svd=(weight_type == 'W_QK'), **config_standard
        )


    def fill(self, layer_idx, head_idx, W_tensor):
            self.histogroup[(layer_idx, head_idx)].fill(W_tensor)


    def post_process(self, metrics=None, do_svd_prob=True):
        if metrics is None:
            from histogram_utils import normality_metrics
            metrics = normality_metrics
        for _, h in self:
            h.stats_values.update({k : v(h) for k, v in metrics.items()})
            h.stats_values['n_entries'] = h.n_entries

    #Persistification
    def to_pandas(self):
        import pandas as pd
        rows = []
        allowed_histos = set()
        for idx, hb in self:
            layer, head = idx
            P_W = hb.histograms['P_W']
            P_l = None
            if 'P_l' in hb.histograms.keys():
                P_l = hb.histograms['P_l']
            row = {
                'layer': layer,
                'head': head,
                'P_W': P_W,
                'P_l' : P_l,
            }
            for k in hb.histograms.keys():
                allowed_histos.add(k)
            if 'SVD' in hb.histograms.keys():
                row['SVD'] = hb.histograms['SVD']
            
            #add stats
            for name, value in hb.stats_values.items():
                if isinstance(value, dict):
                    # Flatten nested dict in stats dict
                    for sub_name, sub_value in value.items():
                        row[sub_name] = sub_value
                else:
                    row[name] = value

            rows.append(row)
        df = pd.DataFrame(rows)
        #now the metadata
        metadata = {}
        metadata['w_bins'] = self.w_bins.tolist()
        metadata['sv_bins'] = self.sv_bins.tolist()
        metadata['histos'] = [hname for hname in allowed_histos]
        return df, metadata

    def load_dataset_with_metadata(path):
        dataset = load_from_disk(path)
        metadata_file = Path(path) / dataset.info.description
        metadata = json.load(open(metadata_file))
        return dataset, metadata


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
        pass
        # stats_config = {'mean' : np.mean}
        # h1 = HistogramBase(bins=bins, stats=stats_config)
        # flat = data.ravel()
        # h1.fill(flat)
        # h1.show()



