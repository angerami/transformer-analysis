from scipy.stats import skew, kurtosis
from scipy.stats import norm, entropy
from scipy.optimize import curve_fit
import numpy as np
from histogram_tools import HistogramGroup

stats_config_standard = {'sum' : np.sum, 'mean' : np.mean, 'std' : np.std, 
                         'max' : np.max, 'min' : np.min, 'skew': skew, 'kurtosis': kurtosis
                         }

weight_bins_standard = np.linspace(-1.6, 1.6, 512)

def build_group_standard(**kwargs):
    return HistogramGroup(stats=stats_config_standard, bins=weight_bins_standard, **kwargs)

def load_group_from_file(filename):
    g = HistogramGroup()
    g.load(filename)
    return g

def dict_to_flat(as_dict):
    return [v for k, v in sorted(as_dict.items())]

def entropy_stat(h):
    p = h.hist_norm()
    return entropy(p)

def kl_vs_standard_normal(h):
    p = h.hist_norm()
    q = norm.pdf(h.get_bin_centers(), 0, 1)
    return entropy(p, q)

def kl_vs_empirical_normal(h):
    mu, sigma = h.get_statistic('mean'), h.get_statistic('std')
    p = h.hist_norm()
    q = norm.pdf(h.get_bin_centers(), mu, sigma)
    return entropy(p, q)

def kl_normal_vs_standard(h):
    mu, sigma = h.get_statistic('mean'), h.get_statistic('std')
    return 0.5 * (sigma**2 + mu**2 - 1 - np.log(sigma**2))

def fit_normal(h, n_sigma=1.5):
    mu, sigma = h.get_statistic('mean'), h.get_statistic('std')
    bin_centers = h.get_bin_centers()
    mask = np.abs(bin_centers - mu) <= n_sigma * sigma
    p = h.hist_norm()

    def gaussian(x, mu, sigma):
        return norm.pdf(x, mu, sigma)
    
    popt, _ = curve_fit(gaussian, bin_centers[mask], p[mask], p0=[mu, sigma])
    return {'fit_mu': popt[0], 'fit_sigma': popt[1]}

normality_metrics = {
    "entropy" : entropy_stat,
    "kl_vs_standard_normal" : kl_vs_standard_normal,
    "kl_vs_empirical_normal" : kl_vs_empirical_normal,
    "kl_normal_vs_standard" : kl_normal_vs_standard,
    "fit_normal" : fit_normal
}

def extract_metrics(h, metrics=normality_metrics):
    return {k : v(h) for k, v in metrics.items()}

def extract_metrics_(h, metrics=normality_metrics):
    h.stats_values.update({k : v(h) for k, v in metrics.items()})


def to_dataframe(hgroup):
    import pandas as pd
    rows = []
    allowed_histos = set()
    for idx, hb in hgroup:
        layer, head = idx
        P_W = hb.histograms['P_W']
        # P_lambda = item.histograms['h']
        
        row = {
            'layer': layer,
            'head': head,
            'P_W': P_W,
        }
        for k in hb.histograms.keys():
            allowed_histos.add(k)
        if 'SVD' in hb.histograms.keys():
            row['SVD'] = hb.histograms['SVD']
        

        for name, value in hb.stats_values.items():
            if isinstance(value, dict):
                # Flatten nested dict
                for sub_name, sub_value in value.items():
                    row[sub_name] = sub_value
            else:
                row[name] = value
        rows.append(row)
        df = pd.DataFrame(rows)
    df.attrs['bins'] = [b for b in hgroup.bins]
    df.attrs['histos'] = [hname for hname in allowed_histos]
    return df

if __name__ == '__main__':

    def MY_TEST_MESSAGE(test_name, result, condition):

        condition = 'PASSED' if condition else 'FAILED'
        print('-'*40 + f'\n{test_name} Test: {result} \n{condition}\n')
    
    def HIST_DIFF(h1, h2):
        return np.sum(np.abs(h1 - h2))

    np.random.seed(123)
    n_layers, n_heads, n_samp = 2, 3, 1000
    bins = np.linspace(-2, 2, 20)
    data = np.random.normal(size=n_layers*n_heads*n_samp).reshape((n_layers, n_heads,n_samp))
    
    g = build_group_standard(n_layers=n_layers, n_heads=n_heads)
    
    for k, h in g:
        h.fill(data[k])
        print(k, f"KL vs standard normal {kl_vs_standard_normal(h):.3f}")
        print(k, f"KL vs empirical {kl_vs_empirical_normal(h):.3f}")
        print(k, f"KL normal vs standard {kl_normal_vs_standard(h):.3f}")
        print(k, "Fit normal ", fit_normal(h, n_sigma=1.5))