import sys

from .csv_dataset import CSVDataset
from .utils import smiles_to_bigraph
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

    Parameters
    ----------
    smiles_to_graph: callable, str -> DGLGraph
        A function turning smiles into a DGLGraph.
        Default to :func:`dgl.data.chem.smiles_to_bigraph`.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.
    """
    def __init__(self, smiles_to_graph=smiles_to_bigraph,
                 atom_featurizer=None,
                 bond_featurizer=None):
        if 'pandas' not in sys.modules:
            from ...base import dgl_warning
            dgl_warning("Please install pandas")

        self._url = 'dataset/delaney.csv'
        data_path = get_download_dir() + '/delaney.csv'
        download(_get_dgl_url(self._url), path=data_path)
        df = pd.read_csv(data_path)

        super().__init__(df, smiles_to_graph, atom_featurizer, bond_featurizer,
                         "smiles", "esol_dglgraph.pkl")
