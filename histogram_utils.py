from scipy.stats import skew, kurtosis
from scipy.stats import norm, entropy
from scipy.optimize import curve_fit
import numpy as np

stats_config_default= {
    'sum' : np.sum, 'mean' : np.mean, 'std' : np.std, 
    'max' : np.max, 'min' : np.min, 'skew': skew, 'kurtosis': kurtosis
}

weight_bins_default = np.linspace(-1.6, 1.6, 512)
sv_bins_default = np.linspace(0,400,400)

bins_dict_default = {
    'w_bins' : weight_bins_default, 
    'sv_bins' : sv_bins_default,
}

config_default = {
    'bins_dict' : bins_dict_default,
    'stats' : {},
    'use_density' : True
}

config_standard = {
    'bins_dict' : bins_dict_default,
    'stats' : stats_config_default,
    'use_density' : True
}



def set_binning_from_dict(h, **kwargs):
    if kwargs and 'bins' in kwargs.keys():
        out_dict = {k : v for k, v in kwargs[bins]}
    else:
        out_dict =  {'w_bins' : weight_bins_default, 'sv_bins' : sv_bins_default}
    h.bin_dict = out_dict
    for k,v in h.bin_dict.items():
        setattr(h, k, v)


def entropy_stat(h, centers=None):
    p = h['P_w']
    h.update({'entropy' : entropy(p)})

def kl_vs_standard_normal(h, centers):
    p = h['P_w']
    q = norm.pdf(centers, 0, 1)
    h.update({'kl_vs_standard_normal' : entropy(p, q)})

def kl_vs_empirical_normal(h, centers):
    mu, sigma = h['mean'], h['std']
    p = h['P_w']
    q = norm.pdf(centers, mu, sigma)
    h.update({'kl_vs_empirical_normal' : entropy(p, q)})

def kl_normal_vs_standard(h, centers):
    mu, sigma = h['mean'], h['std']
    h.update({'kl_vs_empirical_normal' : 
              0.5 * (sigma**2 + mu**2 - 1 - np.log(sigma**2))})

def fit_normal(h, centers, n_sigma=1.5):
    p = h['P_w']
    mu, sigma = h['mean'], h['std']
    if np.isnan(mu) or np.isnan(sigma):
         h.update({'fit_mu': np.nan, 'fit_sigma': np.nan})
    mask = np.abs(centers - mu) <= n_sigma * sigma

    def gaussian(x, mu, sigma):
        return norm.pdf(x, mu, sigma)
    
    popt, _ = curve_fit(gaussian, centers[mask], p[mask], p0=[mu, sigma])
    h.update({'fit_mu': popt[0], 'fit_sigma': popt[1]})


normality_metrics = {
    "entropy" : entropy_stat,
    "kl_vs_standard_normal" : kl_vs_standard_normal,
    "kl_vs_empirical_normal" : kl_vs_empirical_normal,
    "kl_normal_vs_standard" : kl_normal_vs_standard,
    "fit_normal" : fit_normal
}
def extract_metrics(h, metrics=normality_metrics):
    return {k : v(h) for k, v in metrics.items()}



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