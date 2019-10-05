import sys

from .csv_dataset import CSVDataset
from ..utils import get_download_dir, download, _get_dgl_url

try:
    import pandas as pd
except ImportError:
    pass

class ESOL(CSVDataset):
    """ESOL dataset.

    ESOL is a small dataset consisting of water solubility data for 1128 compounds.
    The dataset has been used to train models that estimate solubility directly from
    chemical structures (as encoded in SMILES strings). Note that these structures
    don’t include 3D coordinates, since solubility is a property of a molecule and
    not of its particular conformers.

    All molecules are converted into DGLGraphs. After the first-time construction,
    the DGLGraphs will be saved for reloading so that we do not need to reconstruct them everytime.
    """
    def __init__(self):
        if 'pandas' not in sys.modules:
            from ...base import dgl_warning
            dgl_warning("Please install pandas")

        self._url = 'dataset/delaney.csv'
        data_path = get_download_dir() + '/delaney.csv'
        download(_get_dgl_url(self._url), path=data_path)
        df = pd.read_csv(data_path)
