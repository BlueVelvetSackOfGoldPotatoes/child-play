"""Generate first-paper tables, plots, provenance, and optionally its PDF."""

import argparse
import hashlib
import json
import subprocess
import platform
from importlib.metadata import version
from pathlib import Path

from utils.paper_dataset import ROOT, OUTPUT, TASKS, CONDITIONS, LABELS, analyze
from utils import plot_performance_per_model_best_temp as comparisons
from utils import plot_fresh_outcome_lines as outcomes
from utils import plot_fresh_heatmaps as heatmaps


def table(headers, rows, caption, label):
    """Use the repository's booktabs/resizebox manuscript table format."""
    body = ['\\begin{table}[tbp]', '\\centering', '\\small', '\\resizebox{\\textwidth}{!}{%',
            '\\begin{tabular}{l' + 'r' * (len(headers) - 1) + '}', '\\toprule',
            ' & '.join(headers) + r' \\', '\\midrule']
    body.extend(' & '.join(str(cell) for cell in row) + r' \\' for row in rows)
    body.extend(['\\bottomrule', '\\end{tabular}}', '\\caption{' + caption + '}', '\\label{' + label + '}', '\\end{table}'])
    return '\n'.join(body) + '\n'


def percent(value):
    return f'{100 * value:.2f}'


def write_tables(report):
    metrics = report['metrics']
    rows = [[block['label']] + [percent(block['tasks'][task]['score']) for task in TASKS] + [percent(block['overall'])] for block in metrics.values()]
    (OUTPUT / 'scores.tex').write_text(table(
        ['Condition', 'BS', 'TTT', 'C4', 'Shapes', 'LCL valid.', 'LCL gen.', 'GtS', 'Mean'], rows,
        'Observed success percentages for every evaluated condition. Shapes uses 300 trials; each other task uses 100. The mean gives each task equal weight and includes invalid replies as failures in every task. Temperature is shown in parentheses where supplied. No setting is selected separately for individual tasks.', 'tab:scores'))
    rows = []
    for block in metrics.values():
        gts = block['tasks']['gts']
        rows.append([block['label'], gts['successes'], gts['n'] - gts['successes'] - gts['invalid'], gts['invalid'], percent(gts['score']), percent(gts['conditional_accuracy']) if gts['conditional_accuracy'] is not None else '--', f'{gts["penalized_similarity"]:.3f}'])
    (OUTPUT / 'gts.tex').write_text(table(
        ['Condition', 'Correct', 'Other incorrect', 'Invalid', 'All-reply accuracy', 'Conditional accuracy', 'Penalized similarity'], rows,
        'GtS results from complete retained responses, with 100 drawings per condition. All-reply accuracy is correct/100 and is the primary score. Conditional accuracy excludes the invalid-SMILES sentinel and is secondary. Similarity averages the Dice score with a value of $-1$ for that sentinel; empty or overlength replies instead receive 0. Both accuracy columns are percentages.', 'tab:gts'))
    pieces = []
    for game, title in [('battleship', 'Battleship'), ('tictactoe', 'Tic-Tac-Toe'), ('connectfour', 'Connect-Four')]:
        rows = []
        for block in metrics.values():
            row = block['tasks'][game]
            rows.append([block['label'], row['successes'], row['opponent_wins'], row['ties'], row['invalid']])
        pieces.append(table(['Condition', 'Wins', 'Losses', 'Ties', 'Illegal-reply losses'], rows,
                            title + ' outcomes in 100 games against a random legal opponent. Illegal-reply losses are included in losses, not added to them.', 'tab:' + game))
    (OUTPUT / 'outcomes.tex').write_text('\n'.join(pieces))
    rows = []
    for block in metrics.values():
        row = block['tasks']['shapes']
        rows.append([block['label']] + [percent(row['by_shape'][shape]) for shape in ('square', 'triangle', 'cross')] + [percent(row['score'])])
    (OUTPUT / 'shapes.tex').write_text(table(['Condition', 'Square', 'Triangle', 'Cross', 'Overall'], rows,
        'Shape recognition percentages: 100 boards per class and condition. Overall uses all 300 boards.', 'tab:shapes'))
    macros = {
        'NumRecords': str(report['selected_records']),
        'UniqueMolecules': str(report['molecules']['unique_canonical_graphs']),
        'MinHeavyAtoms': str(report['molecules']['heavy_atoms_min']),
        'MedianHeavyAtoms': str(report['molecules']['heavy_atoms_median']),
        'MaxHeavyAtoms': str(report['molecules']['heavy_atoms_max']),
    }
    for condition, prefix in [('gpt4o-t0', 'FourO'), ('gpt4o-mini-t0', 'FourMini'), ('claude-fable-5-2', 'Fable'), ('claude-opus-5-5', 'Opus'), ('gpt-6-astra', 'Astra'), ('gpt-6.1-sol', 'Sol')]:
        block = metrics[condition]
        macros[prefix + 'Gts'] = str(block['tasks']['gts']['successes'])
        macros[prefix + 'Bs'] = str(block['tasks']['battleship']['successes'])
        macros[prefix + 'Overall'] = percent(block['overall'])
    (OUTPUT / 'summary.tex').write_text('\n'.join('\\newcommand{\\' + key + '}{' + value + '}' for key, value in macros.items()) + '\n')


def render(report):
    write_tables(report)
    comparisons.main(report=report)
    conditions = [(c['id'], LABELS[c['id']]) for c in CONDITIONS]
    metrics = report['metrics']
    for game in ('battleship', 'tictactoe', 'connectfour'):
        counts, errors = {}, {}
        for condition, block in metrics.items():
            row = block['tasks'][game]
            counts[condition] = {'wins': row['successes'], 'random': row['opponent_wins'], 'ties': row['ties'], 'incorrect': row['invalid']}
            errors[condition] = {metric: outcomes._binary_se_percent([1] * count + [0] * (row['n'] - count)) for metric, count in [('Wins', row['successes']), ('Wins (Random Player)', row['opponent_wins']), ('Ties', row['ties']), ('Incorrect Moves', row['invalid'])]}
        outcomes.plot_fresh_outcome(counts, None, errors, OUTPUT / (game + '_outcomes.pdf'), conditions=conditions)
    for game, values in report['tactical'].items():
        outcomes.plot_fresh_missed(values, [], OUTPUT / (game + '_tactics.pdf'), conditions=conditions)
    heatmaps.main(complete=True)
    write_provenance(report)
    print('Rendered current tables, seven comparison plots, outcome/tactical plots, and every move heatmap.')


def write_provenance(report):
    sources = [ROOT / name for name in ('fresh_run.py', 'fresh_gts.py', 'lcl.py', 'wrapper.py', 'main.py', 'requirements-runtime.txt', 'paper/main_cell.tex', 'paper/main.bib', 'paper/childplay_review_references.bib', 'paper/numbered.bst')]
    sources += list((ROOT / 'scripts_games').glob('*.py'))
    sources += [ROOT / 'utils' / name for name in ('paper_dataset.py', 'render_paper.py', 'board_move_metrics.py', 'count_blocking_moves_connect_four.py', 'count_blocking_moves_tictactoe.py', 'plot_fresh_heatmaps.py', 'plot_fresh_outcome_lines.py', 'plot_performance_per_model_best_temp.py')]
    sources += [ROOT / 'molecule_app' / name for name in ('app.py', 'scoring.py', 'make_dataset.py')]
    artifacts = list(OUTPUT.iterdir())
    if (ROOT / 'paper/main_cell.pdf').exists():
        artifacts.append(ROOT / 'paper/main_cell.pdf')
    provenance = {
        'dataset_sources': report['sources'], 'stimulus_sources': report['stimuli'],
        'code_sources': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(sources))},
        'runtime': {'python': platform.python_version(), 'packages': {name: version(name) for name in ('numpy', 'pandas', 'matplotlib', 'seaborn', 'rdkit', 'selfies', 'Flask', 'Jinja2', 'openai')}},
        'artifacts': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(artifacts) if p.suffix in ('.json', '.tex', '.pdf') and p.name != 'provenance.json'},
    }
    (OUTPUT / 'provenance.json').write_text(json.dumps(provenance, indent=2))


def build():
    commands = [['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main_cell.tex'], ['bibtex', 'main_cell'], ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main_cell.tex'], ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main_cell.tex']]
    for index, command in enumerate(commands, 1):
        log = OUTPUT / f'build-{index}.log'
        with log.open('w') as handle:
            subprocess.run(command, cwd=ROOT / 'paper', stdout=handle, stderr=subprocess.STDOUT, check=True)
    print('Built paper/main_cell.pdf with the existing pdflatex/BibTeX pipeline.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='Also build the current first-paper PDF')
    args = parser.parse_args()
    report = analyze()
    render(report)
    if args.build:
        build()
        write_provenance(report)
