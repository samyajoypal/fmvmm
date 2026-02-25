"""Inference subpackage for fmvmm.

Currently includes DMM-focused inference helpers:
- Wald tests (linear constraints)
- Score tests for fixed-parameter hypotheses
- Likelihood ratio tests (regular cases)

See :mod:`fmvmm.inference.inference_dmm`.
"""

from .inference_dmm import (
    WaldTestResult,
    ScoreTestResult,
    LRTResult,
    wald_test,
    wald_test_alpha_equal,
    wald_test_eta_equal,
    score_test_fixed,
    lrt,
    dims_from_model,
    idx_eta,
    idx_alpha,
    build_test_indices_alpha,
    build_test_indices_eta,
)

__all__ = [
    "WaldTestResult",
    "ScoreTestResult",
    "LRTResult",
    "wald_test",
    "wald_test_alpha_equal",
    "wald_test_eta_equal",
    "score_test_fixed",
    "lrt",
    "dims_from_model",
    "idx_eta",
    "idx_alpha",
    "build_test_indices_alpha",
    "build_test_indices_eta",
]
