
# Organizational matters
Three public facing objects:
- Github `angerami/angerami.github.io`; "website repository"; Jekyll Academic Pages
        https://angerami.github.io
    - Posts section : contains relevant/linked posts describing theory, background and motivation
    - Projects section : Will eventually contain material presenting a project's results and linking to the project repo and hugging face Space; 
    - Other sections (bio, CV, publications) not relevant for current chat
- Github `angerami/transformer-analysis`; "Project repository"
    - Project code
    - Readme describing project and linking to other entities
    - A streamlit app that is a template or prototype of the one to appear on Hugging Face
    - Data artifacts (pkl files) or use dedicated dataset provider (zenodo or is that only for publication?)
- Hugging Face Space `angerami/transformer-spin-explorer`
    - Streamlit app (copy of one in project repo?)
    - Data artifacts

Question of how to treat this
-  approach 1: website "projects" are not a home for projects (thats what the github page is for), its just a single page that has highlight content

- approach 2: As my website has posts that explore the material it might be really natural to create objects directly from cleaned up jupyter sessions that effectively describe the project and could make good content as posts. in theory it could be accomplished by a straightforward distillation process of jupyter notebooks that are in the project files, or would a website post possibly just have an ipynb file of the same name that goes a long with it as part of the post itself? how do we handle the fact that the notebook code will depend on my own library? or does that give me an opportunity to have other people clone my repo and provides a bonus self-promotion mechanism?



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

## `EleutherAI/pythia-70m-deduped`
```
GPTNeoXForCausalLM(
  (gpt_neox): GPTNeoXModel(
    (embed_in): Embedding(50304, 512)
    (emb_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-5): 6 x GPTNeoXLayer(
        (input_layernorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (post_attention_layernorm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (post_attention_dropout): Dropout(p=0.0, inplace=False)
        (post_mlp_dropout): Dropout(p=0.0, inplace=False)
        (attention): GPTNeoXAttention(
          (query_key_value): Linear(in_features=512, out_features=1536, bias=True)
          (dense): Linear(in_features=512, out_features=512, bias=True)
        )
        (mlp): GPTNeoXMLP(
          (dense_h_to_4h): Linear(in_features=512, out_features=2048, bias=True)
          (dense_4h_to_h): Linear(in_features=2048, out_features=512, bias=True)
          (act): GELUActivation()
        )
      )
    )
    (final_layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    (rotary_emb): GPTNeoXRotaryEmbedding()
  )
  (embed_out): Linear(in_features=512, out_features=50304, bias=False)
   ```

