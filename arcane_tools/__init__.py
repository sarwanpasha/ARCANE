"""arcane_tools: scvi-tools-compatible API for ARCANE label transfer with error control.

>>> from arcane_tools import ARCANE
>>> model = ARCANE.fit(adata_ref, batch_key="donor_id", labels_key="cell_type")
>>> model.map(adata_query, control_key="disease", control_value="normal")

See ``api.ARCANE`` for parameters and ``tutorial/lupus_tutorial.md`` for a walk-through.
"""
from .api import ARCANE, ArcaneWarning, control_recal, greedy_folds

__all__ = ["ARCANE", "ArcaneWarning", "control_recal", "greedy_folds"]
__version__ = "0.1.0"
