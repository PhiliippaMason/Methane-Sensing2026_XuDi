"""
radtran.py

Placeholder/interface for the radiative transfer model component used in
methane column concentration estimation.

Important
---------
The original radiative transfer model code referenced in:

    "High-frequency monitoring of anomalous methane point sources with
    multispectral Sentinel-2 satellite observations"

is not included in this repository.

The referenced study states:

    "The radiative transfer model code used to calculate methane column
    concentrations will be made available upon request."

Users who need the original implementation should contact the authors of the
referenced study directly.

This file does not reproduce, redistribute, or claim ownership of the original
radiative transfer model code. It only provides a clear integration point for
an authorized or independently developed radiative transfer model.

Suggested Use
-------------
Replace the placeholder function below with:

1. an authorized implementation obtained from the original authors; or
2. an independently developed and validated radiative transfer model.

Do not use this placeholder as a scientific methane column concentration model.
"""


def retrieve(frac, instr, m, tar, obs, solar, ob, n):
    """
    RTM model
    """
    return True
