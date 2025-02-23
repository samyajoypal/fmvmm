# FMVMM: Flexible Multivariate Mixture Model

FMVMM is a Python package providing a comprehensive collection of finite mixture models for various multivariate distributions. It is designed to offer flexibility in modeling both identical and non-identical mixture distributions, allowing users to apply advanced clustering techniques efficiently.

## Features
- Implements finite mixtures of various multivariate distributions.
- Supports clustering using Dirichlet Mixture Models and mixtures of other non-identical distributions.
- Provides multiple model selection criteria such as AIC, BIC, and ICL.
- Includes efficient parameter estimation and standard error calculations.
- Example Jupyter notebooks available in the `examples/` folder.

## Mixture Models Included
- **Dirichlet Mixture Model** (`DMM_Soft`, `DMM_Hard`)
- **Mixtures of Multivariate Generalized Hyperbolic Distributions** (`MixMGH`)
- **Mixtures of Skew Normal Distributions** (`SkewNormalMix`)
- **Mixtures of Skew T Distributions** (`SkewTMix`)
- **Mixtures of T Distributions** (`TMix`)
- **Mixtures of Skew Normal Contaminated Distributions** (`SkewContMix`)
- **Mixtures of Skew Slash Distributions** (`SkewSlashMix`)
- **Mixtures of Slash Distributions** (`SlashMix`)
- **Flexible Multivariate Mixture Model** (`FMVMM`)
  - This model allows fitting all possible combinations of mixtures of different identical and non-identical distributions.

## Distributions Included
- Multivariate Skew Normal
- Multivariate Normal
- Multivariate T
- Multivariate Generalized Skew T
- Multivariate Skew T
- Multivariate Hyperbolic
- Multivariate Normal Inverse Gaussian
- Multivariate Variance Gamma
- Multivariate Skew Slash
- Multivariate Slash
- Multivariate Skew Normal Contaminated

## Installation
You can install FMVMM using:
```bash
pip install fmvmm
```

## Usage
Each Python class in FMVMM provides the following methods:

```python
from fmvmm.mixtures.DMM_Soft import DMM_Soft

model = DMM_Soft(n_clusters=3)
model.fit(data)  # Fit a mixture model
clusters = model.predict()  # Get cluster assignments
bic_value = model.bic()  # Compute Bayesian Information Criterion
aic_value = model.aic()  # Compute Akaike Information Criterion
icl_value = model.icl()  # Compute Integrated Complete Likelihood Criterion
pi, alpha = model.get_params()  # Get MLE of parameters
info_matrix, se = model.get_info_mat()  # Get information matrix and standard errors
```

For more detailed examples, see the Jupyter notebooks in the `examples/` folder.

## Citation
If you use FMVMM in your research, please cite the relevant papers:

1. Pal, Samyajoy, and Christian Heumann. "Clustering compositional data using Dirichlet mixture model." *PLoS ONE* 17, no. 5 (2022): e0268438. [https://doi.org/10.1371/journal.pone.0268438](https://doi.org/10.1371/journal.pone.0268438).
2. Pal, Samyajoy, and Christian Heumann. "Gene coexpression analysis with Dirichlet mixture model: accelerating model evaluation through closed-form KL divergence approximation using variational techniques." *International Workshop on Statistical Modelling* (2024). [https://doi.org/10.1007/978-3-031-65723-8_21](https://doi.org/10.1007/978-3-031-65723-8_21).
3. Pal, Samyajoy, and Christian Heumann. "Revisiting Dirichlet Mixture Model: Unraveling Deeper Insights and Practical Applications." *Statistical Papers* 66, no. 1 (2025): 1-38. [https://doi.org/10.1007/s00362-024-01627-0](https://doi.org/10.1007/s00362-024-01627-0).
4. Pal, Samyajoy, and Christian Heumann. "Flexible Multivariate Mixture Models: A Comprehensive Approach for Modeling Mixtures of Non-Identical Distributions." *International Statistical Review* (2024). [https://doi.org/10.1111/insr.12593](https://doi.org/10.1111/insr.12593).

## License
FMVMM is open-source and distributed under the MIT License.

## Contact
For any questions or contributions, feel free to reach out or open an issue on GitHub.
