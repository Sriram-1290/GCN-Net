"""The 64 EEGMMIDB channel names, shared by tests that need only the names.

Hard-coded in cache order (standardised by ``mne.datasets.eegbci.standardize``) so those
tests run on a fresh clone without ``data/cache_bp830``.
``tests/test_interpolate.py::test_channel_names_match_the_cache`` asserts the list equals
what the cache holds whenever the cache exists, so the two cannot drift silently.
"""

CH_NAMES = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "Fp1", "Fpz", "Fp2", "AF7",
    "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "Pz",
    "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
)
