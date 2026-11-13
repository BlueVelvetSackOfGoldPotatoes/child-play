"""Reproduce the first paper from complete model-response ledgers.

This is the publication analysis entry point. It reuses the benchmark's actual
parsers, game replay, LCL checker, and molecular scorer; it makes no model calls.
"""

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fresh_gts import extract_smiles, score_prediction
from fresh_run import CONDITIONS, AVAILABLE_LEGACY_CONDITIONS, _parse_pieces, parse_int
from lcl import LCLGame
from utils.plot_fresh_heatmaps import load_rows, load_stimuli, replay
from utils.plot_fresh_outcome_lines import fresh_logs, fresh_missed

CONDITIONS = AVAILABLE_LEGACY_CONDITIONS + CONDITIONS
LABELS = {
    'gpt4o-t0': 'GPT-4o (0)', 'gpt4o-mini-t0': 'GPT-4o-mini (0)',
    'claude-fable-5-2': 'Fable 5.2', 'claude-opus-5-5': 'Opus 5.5',
    'gpt-6-astra': 'Astra', 'gpt-6.1-sol': 'Sol',
    'gpt-6-luna-t0': 'Luna (0)', 'gpt-6-luna-t0.5': 'Luna (0.5)',
    'gpt-6-luna-t1': 'Luna (1)',
}
TASKS = ('battleship', 'tictactoe', 'connectfour', 'shapes', 'lcl_validity', 'lcl_construct', 'gts')
SOURCES = (
    'fresh_experiments/seed20261111/results.jsonl',
    'fresh_experiments/seed20261111_bs_clause/results.jsonl',
    'fresh_experiments/seed20261111_complete/gts_results.jsonl',
    'fresh_experiments/seed20261111_legacy_complete/results.jsonl',
    'fresh_experiments/seed20261111_legacy_complete/gts_results.jsonl',
    'fresh_experiments/seed20261111_lcl_rules/results.jsonl',
)
OUTPUT = ROOT / 'paper' / 'complete'


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_dataset():
    """Load the selected complete cohort without altering historical records."""
    rows = load_rows()
    for name in SOURCES[2:5]:
        rows.extend(read_jsonl(ROOT / name))
    rows = [row for row in rows if row['task'] not in ('lcl_validity', 'lcl_construct')]
    rows.extend(read_jsonl(ROOT / SOURCES[5]))
    return rows


def expected_ids(task):
    if task == 'shapes':
        return {f'{shape}-{i}' for shape in ('square', 'triangle', 'cross') for i in range(100)}
    if task == 'gts':
        return {f'gts-{i}' for i in range(100)}
    return set(range(100))


def analyze():
    """Re-score every selected real record and emit publication metrics."""
    rows = load_dataset()
    conditions = {c['id']: c for c in CONDITIONS}
    index = {}
    for row in rows:
        key = (row['condition'], row['task'], row['episode'])
        if key in index:
            raise ValueError(f'duplicate result: {key}')
        if row['condition'] not in conditions or row['task'] not in TASKS:
            raise ValueError(f'unexpected result: {key}')
        condition = conditions[row['condition']]
        if row.get('model') != condition['model'] or row.get('temperature') != condition['temperature']:
            raise ValueError(f'model configuration mismatch: {key}')
        if row.get('error'):
            raise ValueError(f'incomplete model call: {key}: {row["error"]}')
        index[key] = row
    for condition in conditions:
        for task in TASKS:
            actual = {k[2] for k in index if k[:2] == (condition, task)}
            if actual != expected_ids(task):
                raise ValueError(f'incomplete {condition}/{task}: missing={expected_ids(task)-actual}, extra={actual-expected_ids(task)}')
    stimuli = load_stimuli()
    clause = json.loads((ROOT / 'fresh_experiments/seed20261111_bs_clause/stimuli.json').read_text())
    legacy = json.loads((ROOT / 'fresh_experiments/seed20261111_legacy_complete/stimuli.json').read_text())
    lcl = json.loads((ROOT / 'fresh_experiments/seed20261111_lcl_rules/stimuli.json').read_text())
    if legacy != clause:
        raise ValueError('the older-model cohort did not use the corrected shared stimuli')
    for name, candidate in [('Battleship clause', clause), ('explicit LCL', lcl)]:
        if {k: v for k, v in candidate.items() if k != 'prompts'} != {k: v for k, v in stimuli.items() if k != 'prompts'}:
            raise ValueError(f'{name} changed an input item or opponent seed')
    for task, prompt in stimuli['prompts'].items():
        if task != 'battleship' and clause['prompts'][task] != prompt:
            raise ValueError(f'Battleship clause changed another prompt: {task}')
    for task, prompt in stimuli['prompts'].items():
        if task not in ('lcl_validity', 'lcl_construct') and lcl['prompts'][task] != prompt:
            raise ValueError(f'LCL revision changed another prompt: {task}')
    episodes = {task: {item['id']: item for item in stimuli[task]} for task in TASKS if task != 'gts'}
    molecules = json.loads((ROOT / 'fresh_experiments/seed20261111/gts_stimuli.json').read_text())['molecules']
    molecules_by_id = {m['id']: m for m in molecules}
    checker = LCLGame()
    for item in episodes['lcl_validity'].values():
        pieces = [tuple(piece) for piece in item['pieces']]
        if checker.is_valid_construct(pieces) != item['valid']:
            raise ValueError(f'LCL stimulus label mismatch: {item["id"]}')
    for key, row in index.items():
        condition, task, episode = key
        if task in ('battleship', 'tictactoe', 'connectfour'):
            accepted, _, winner = replay(task, episodes[task][episode], row['moves'])
            if winner != row['winner'] or (len(accepted) < len(row['moves'])) != row['invalid']:
                raise ValueError(f'gameplay mismatch: {key}')
            continue
        if task == 'gts':
            if 'response' not in row or 'predicted' not in row:
                raise ValueError(f'missing complete model reply: {key}')
            if extract_smiles(row['response']) != row['predicted']:
                raise ValueError(f'reply extraction mismatch: {key}')
            scored = score_prediction(molecules_by_id[episode]['smiles'], row['predicted'])
            for field, value in scored.items():
                stored = row[field]
                equal = math.isclose(stored, value, abs_tol=1e-12) if type(value) is float else stored == value
                if not equal:
                    raise ValueError(f'molecular scoring mismatch: {key}/{field}')
            continue
        text = row['moves'][0]
        if task == 'shapes':
            item = episodes[task][episode]
            choice = parse_int(text)
            invalid = choice is None or not 0 <= choice < len(item['options'])
            correct = not invalid and item['options'][choice] == item['shape']
        elif task == 'lcl_validity':
            invalid = text.strip().lower() not in ('valid', 'invalid')
            correct = not invalid and (text.strip().lower() == 'valid') == episodes[task][episode]['valid']
        else:
            pieces = _parse_pieces(text)
            invalid = pieces is None
            correct = not invalid and len(pieces) == 3 and checker.is_valid_construct(pieces)
        if bool(correct) != row['correct'] or invalid != row['invalid']:
            raise ValueError(f'task scoring mismatch: {key}')
    metrics = {}
    for condition in conditions:
        task_metrics = {}
        for task in TASKS:
            block = [row for key, row in index.items() if key[:2] == (condition, task)]
            n = len(block)
            success = sum(row['winner'] == 'P1' if task in ('battleship', 'tictactoe', 'connectfour') else bool(row['correct']) for row in block)
            record = {'n': n, 'successes': success, 'score': success / n, 'invalid': sum(bool(row['invalid']) for row in block)}
            if task in ('battleship', 'tictactoe', 'connectfour'):
                record.update(opponent_wins=sum(row['winner'] == 'P2' for row in block), ties=sum(row['winner'] == 'tie' for row in block))
            if task == 'gts':
                record['penalized_similarity'] = sum(row['chemical_similarity'] for row in block) / n
                graded = n - record['invalid']
                record['conditional_accuracy'] = success / graded if graded else None
            if task == 'shapes':
                record['by_shape'] = {shape: sum(row['correct'] for row in block if str(row['episode']).startswith(shape + '-')) / 100 for shape in ('square', 'triangle', 'cross')}
            task_metrics[task] = record
        metrics[condition] = {'label': LABELS[condition], 'model': conditions[condition]['model'], 'temperature': conditions[condition]['temperature'], 'tasks': task_metrics, 'overall': sum(task_metrics[t]['score'] for t in TASKS) / len(TASKS)}
    tactical = {game: fresh_missed(game, fresh_logs(game, rows, episodes[game])) for game in ('tictactoe', 'connectfour')}
    from rdkit import Chem
    sizes = sorted(Chem.MolFromSmiles(m['smiles']).GetNumAtoms() for m in molecules)
    unique = len({Chem.MolToSmiles(Chem.MolFromSmiles(m['smiles'])) for m in molecules})
    report = {
        'seed': 20261111, 'selected_records': len(rows), 'conditions': len(conditions),
        'outcome_and_scoring_mismatches': 0, 'metrics': metrics, 'tactical': tactical,
        'molecules': {'draws': len(molecules), 'unique_canonical_graphs': unique, 'heavy_atoms_min': sizes[0], 'heavy_atoms_median': (sizes[49] + sizes[50]) / 2, 'heavy_atoms_max': sizes[-1]},
        'sources': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
        'stimuli': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('fresh_experiments/seed20261111/stimuli.json', 'fresh_experiments/seed20261111_bs_clause/stimuli.json', 'fresh_experiments/seed20261111/gts_stimuli.json', 'fresh_experiments/seed20261111_legacy_complete/stimuli.json', 'fresh_experiments/seed20261111_lcl_rules/stimuli.json')},
        'selection': 'All retained complete current comparisons; GtS is a full new run. Historical incomplete/retired-model runs are preserved and excluded from this comparison.'
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(f'Rescored {len(rows)} real model records across {len(conditions)} complete conditions; no mismatches.')
    for condition, block in metrics.items():
        print(condition, {task: round(block['tasks'][task]['score'] * 100, 2) for task in TASKS}, 'overall', round(block['overall'] * 100, 2))
    print('Molecular stimulus content:', report['molecules'])
    return report


if __name__ == '__main__':
    analyze()
