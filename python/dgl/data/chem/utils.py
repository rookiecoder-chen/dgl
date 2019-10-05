import dgl.backend as F
import itertools
import numpy as np
from functools import partial

from dgl import DGLGraph

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolfiles, rdmolops
except ImportError:
    pass

__all__ = ['one_hot_encoding', 'atom_type_one_hot', 'atom_degree_one_hot',
           'atom_implicit_valence_one_hot', 'atom_hybridization_one_hot',
           'atom_total_num_H_one_hot', 'atom_formal_charge', 'atom_num_radical_electrons',
           'atom_is_aromatic', 'BaseAtomFeaturizer', 'ConcatAtomFeaturizer',
           'CanonicalAtomFeaturizer', 'mol_to_graph', 'smiles_to_bigraph', 'mol_to_bigraph',
           'smiles_to_complete_graph', 'mol_to_complete_graph']

def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """One-hot encoding.

    Parameters
    ----------
    x : str, int or Chem.rdchem.HybridizationType
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        last element.

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if (x not in allowable_set) and encode_unknown:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_type_one_hot(atom, allowable_atom_types=None, encode_unknown=False):
    """One hot encoding for the type of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_atom_types : list of str
        Atom types to consider. Default: ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``,
        ``Cl``, ``Br``, ``Mg``, ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``,
        ``K``, ``Tl``, ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
        ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``, ``Cr``,
        ``Pt``, ``Hg``, ``Pb``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_atom_types is None:
        allowable_atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                                'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                                'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                                'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    return one_hot_encoding(atom.GetSymbol(), allowable_atom_types, encode_unknown)

def atom_degree_one_hot(atom, allowable_atom_degrees=None, encode_unknown=False):
    """One hot encoding for the degree of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_atom_degrees : list of int
        Atom degrees to consider. Default: ``0`` - ``10``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_atom_degrees is None:
        allowable_atom_degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return one_hot_encoding(atom.GetDegree(), allowable_atom_degrees, encode_unknown)

def atom_implicit_valence_one_hot(atom, allowable_atom_valences=None, encode_unknown=False):
    """One hot encoding for the implicit valences of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_atom_valences : list of int
        Atom implicit valences to consider. Default: ``0`` - ``6``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_atom_valences is None:
        allowable_atom_valences = [0, 1, 2, 3, 4, 5, 6]
    return one_hot_encoding(atom.GetImplicitValence(), allowable_atom_valences, encode_unknown)

def atom_hybridization_one_hot(atom, allowable_atom_hybridizations=None, encode_unknown=False):
    """One hot encoding for the hybridization of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_atom_hybridizations : list of rdkit.Chem.rdchem.HybridizationType
        Atom hybridizations to consider. Default: ``Chem.rdchem.HybridizationType.SP``,
        ``Chem.rdchem.HybridizationType.SP2``, ``Chem.rdchem.HybridizationType.SP3``,
        ``Chem.rdchem.HybridizationType.SP3D``, ``Chem.rdchem.HybridizationType.SP3D2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_atom_hybridizations is None:
        allowable_atom_hybridizations = [Chem.rdchem.HybridizationType.SP,
                                         Chem.rdchem.HybridizationType.SP2,
                                         Chem.rdchem.HybridizationType.SP3,
                                         Chem.rdchem.HybridizationType.SP3D,
                                         Chem.rdchem.HybridizationType.SP3D2]
    return one_hot_encoding(atom.GetHybridization(), allowable_atom_hybridizations, encode_unknown)

def atom_total_num_H_one_hot(atom, allowable_atom_num_H=None, encode_unknown=False):
    """One hot encoding for the total number of Hs of an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_atom_num_H : list of int
        Total number of Hs to consider. Default: ``0`` - ``4``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        last element. (Default: False)

    Returns
    -------
    list
        List of boolean values where at most one value is True.
    """
    if allowable_atom_num_H is None:
        allowable_atom_num_H = [0, 1, 2, 3, 4]
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_atom_num_H, encode_unknown)

def atom_formal_charge(atom):
    """Get formal charge for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.
    """
    return [atom.GetFormalCharge()]

def atom_num_radical_electrons(atom):
    """Get the number of radical electrons for an atom.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one int only.
    """
    return [atom.GetNumRadicalElectrons()]

def atom_is_aromatic(atom):
    """Get whether the atom is aromatic.

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.

    Returns
    -------
    list
        List containing one bool only.
    """
    return [atom.GetIsAromatic()]

class BaseAtomFeaturizer(object):
    """An abstract class for atom featurizers

    All atom featurizers that map a molecule to atom features should subclass it.
    All subclasses should overwrite ``_featurize_atom``, which featurizes a single
    atom and ``__call__``, which loops over all atoms and featurize them with
    ``_featurize_atom``.
    """
    def _featurize_atom(self, atom):
        return NotImplementedError

    def __call__(self, mol):
        return NotImplementedError

class ConcatAtomFeaturizer(BaseAtomFeaturizer):
    """Initialize a single atom feature by concatenating multiple descriptors of it,
    e.g. atom degree.

    Parameters
    ----------
    atom_descriptor_funcs : list
        List of functions for computing descriptors of atoms. Each function is of
        signature ``func(rdkit.Chem.rdchem.Atom) -> list of float or bool or int``.
        The resulting order of the atom feature will follow that of the functions
        in the list.
    atom_data_field : str
        Name for storing atom features in DGLGraphs. Default to be ``'h'``.
    """
    def __init__(self, atom_descriptor_funcs, atom_data_field='h'):
        super(ConcatAtomFeaturizer, self).__init__()

        self.atom_descriptor_funcs = atom_descriptor_funcs
        self.atom_data_field = atom_data_field
        self.atom_feat_size = None

    @property
    def feat_size(self):
        """Returns feature size

        Returns
        -------
        int
        """
        if self.atom_feat_size is None:
            atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
            self.atom_feat_size = len(self._featurize_atom(atom))
        return self.atom_feat_size

    def _featurize_atom(self, atom):
        """Featurize an atom

        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom
            Container for atom information.

        Returns
        -------
        results : list
            List of feature values, which can be of type bool, float or int.
        """
        return list(itertools.chain.from_iterable(
            [func(atom) for func in self.atom_descriptor_funcs]))

    def __call__(self, mol):
        """Featurize a molecule

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Store the atom feature under key ``self.atom_data_field``.
            The feature is tensor of dtype float32 and shape (N, self.atom_feat_size),
            where N is the number of atoms in the molecule
        """
        num_atoms = mol.GetNumAtoms()
        atom_features = []
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_features.append(self._featurize_atom(atom))
        atom_features = np.stack(atom_features)
        atom_features = F.zerocopy_from_numpy(atom_features.astype(np.float32))

        return {self.atom_data_field: atom_features}

class CanonicalAtomFeaturizer(ConcatAtomFeaturizer):
    """A default featurizer for atoms.

    The atom features include:

    * **One hot encoding of the atom type**. The supported atom types include
      ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``, ``Cl``, ``Br``, ``Mg``,
      ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``, ``K``, ``Tl``,
      ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
      ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``,
      ``Cr``, ``Pt``, ``Hg``, ``Pb``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 10``.
    * **One hot encoding of the number of implicit Hs on the atom**. The supported
      possibilities include ``0 - 6``.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to be 'h'.
    """
    def __init__(self, atom_data_field='h'):
        super(CanonicalAtomFeaturizer, self).__init__(
            atom_descriptor_funcs=[
                atom_type_one_hot,
                atom_degree_one_hot,
                atom_implicit_valence_one_hot,
                atom_formal_charge,
                atom_num_radical_electrons,
                atom_hybridization_one_hot,
                atom_is_aromatic,
                atom_total_num_H_one_hot
            ],
            atom_data_field=atom_data_field)

def mol_to_graph(mol, graph_constructor, atom_featurizer, bond_featurizer):
    """Convert an RDKit molecule object into a DGLGraph and featurize for it.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    graph_constructor : callable
        Takes an RDKit molecule as input and returns a DGLGraph
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.

    Returns
    -------
    g : DGLGraph
        Converted DGLGraph for the molecule
    """
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    g = graph_constructor(mol)

    if atom_featurizer is not None:
        g.ndata.update(atom_featurizer(mol))

    if bond_featurizer is not None:
        g.edata.update(bond_featurizer(mol))

    return g

def construct_bigraph_from_mol(mol, add_self_loop=False):
    """Construct a bi-directed DGLGraph with topology only for the molecule.

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.

    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty bigraph topology of the molecule
    """
    g = DGLGraph()

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
    g.add_edges(src_list, dst_list)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)

    return g

def mol_to_bigraph(mol, add_self_loop=False,
                   atom_featurizer=None,
                   bond_featurizer=None):
    """Convert an RDKit molecule object into a bi-directed DGLGraph and featurize for it.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.

    Returns
    -------
    g : DGLGraph
        Bi-directed DGLGraph for the molecule
    """
    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        atom_featurizer, bond_featurizer)

def smiles_to_bigraph(smiles, add_self_loop=False,
                     atom_featurizer=None,
                     bond_featurizer=None):
    """Convert a SMILES into a bi-directed DGLGraph and featurize for it.

    Parameters
    ----------
    smiles : str
        String of SMILES
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.

    Returns
    -------
    g : DGLGraph
        Bi-directed DGLGraph for the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_bigraph(mol, add_self_loop, atom_featurizer, bond_featurizer)

def construct_complete_graph_from_mol(mol, add_self_loop=False):
    """Construct a complete graph with topology only for the molecule

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The edges are in the order of (0, 0), (1, 0), (2, 0), ... (0, 1), (1, 1), (2, 1), ...
    If self loops are not created, we will not have (0, 0), (1, 1), ...

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.

    Returns
    -------
    g : DGLGraph
        Empty complete graph topology of the molecule
    """
    g = DGLGraph()
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    if add_self_loop:
        g.add_edges(
            [i for i in range(num_atoms) for j in range(num_atoms)],
            [j for i in range(num_atoms) for j in range(num_atoms)])
    else:
        g.add_edges(
            [i for i in range(num_atoms) for j in range(num_atoms - 1)], [
                j for i in range(num_atoms)
                for j in range(num_atoms) if i != j
            ])

    return g

def mol_to_complete_graph(mol, add_self_loop=False,
                          atom_featurizer=None,
                          bond_featurizer=None):
    """Convert an RDKit molecule into a complete DGLGraph and featurize for it.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.

    Returns
    -------
    g : DGLGraph
        Complete DGLGraph for the molecule
    """
    return mol_to_graph(mol, partial(construct_complete_graph_from_mol, add_self_loop=add_self_loop),
                        atom_featurizer, bond_featurizer)

def smiles_to_complete_graph(smiles, add_self_loop=False,
                             atom_featurizer=None,
                             bond_featurizer=None):
    """Convert a SMILES into a complete DGLGraph and featurize for it.

    Parameters
    ----------
    smiles : str
        String of SMILES
    add_self_loop : bool
        Whether to add self loops in DGLGraphs. Default to False.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to None.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph. Default to None.

    Returns
    -------
    g : DGLGraph
        Complete DGLGraph for the molecule
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_complete_graph(mol, add_self_loop, atom_featurizer, bond_featurizer)
