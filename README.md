# transformer-analysis
Analysis code for analyzing transformers

## To do
- Improve configuration for `run_weight_analysis.py` 
    - Resource printing and logging
    - file naming
    - add W_Q and W_K separately
- Use something better than homegrown I/O
- Develop plot dashboards
- SVDs - extract single, analyze and develop metrics
- Demos and intuition
    - P(W) ~ N(0, 1)
        - KL divergence
            - D_KL (N(mu, sigma)||N(0,1))
            - D_KL ( P(W) || N(0,1)), where P ~ N(0,1) + other components
        - SVD
            - SV shape
            - SV distribution
    - P(W) for W = Q^TK where Q and K are n x m 
        - Properties:
            - General non-Gaussianity
            - Systematics of shape dependence on n, m, initial mu, sigma
            - KL divergence wrt some underlying Gaussian (?)
        - SVD
            - SV shape
            - SV distribution

            

