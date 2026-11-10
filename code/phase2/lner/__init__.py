"""LNER -- Phase 2 (Legal Named-Entity Recognition) package.

Span prediction with strict (start, end, label) matching on top of the Phase-1 legal language models.
Shared dataset state lives in ``lner.state`` and is filled by ``lner.data.dataset.init_dataset()``.
"""
