"""CJPE (Court Judgment Prediction & Explanation) -- Phase 4 package.

Importing this package first imports ``cjpe.context`` which neutralises a broken
``torchvision`` install (common on Kaggle) before ``transformers`` is imported.
"""
from . import context  # noqa: F401  (must be first: patches torchvision)
