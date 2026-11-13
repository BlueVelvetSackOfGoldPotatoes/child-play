"""Canonical molecular scoring shared by the live API and benchmark runner."""

import Levenshtein
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

MAX_SMILES_LENGTH = 200


def score_prediction(original, predicted):
    """Score canonical graph equality and radius-2 Morgan-count Dice similarity.

    Empty and overlength replies are incorrect with similarity 0. Parser
    rejection receives the historical -1 sentinel. Invalid targets and
    unexpected fingerprinting errors propagate as data/runtime failures.
    """
    canonical_original = Chem.CanonSmiles(original)
    if not predicted or len(predicted) > MAX_SMILES_LENGTH:
        return {'correct': False, 'chemical_similarity': 0.0, 'string_distance': -1, 'invalid': False}
    right = Chem.MolFromSmiles(predicted)
    if right is None:
        return {'correct': False, 'chemical_similarity': -1.0,
                'string_distance': int(Levenshtein.distance(original, predicted)), 'invalid': True}
    canonical_predicted = Chem.MolToSmiles(right)
    left = Chem.MolFromSmiles(canonical_original)
    generator = AllChem.GetMorganGenerator(radius=2)
    similarity = float(DataStructs.DiceSimilarity(
        generator.GetSparseCountFingerprint(left), generator.GetSparseCountFingerprint(right)))
    return {'correct': canonical_original == canonical_predicted,
            'chemical_similarity': similarity,
            'string_distance': int(Levenshtein.distance(canonical_original, canonical_predicted)),
            'invalid': False}
