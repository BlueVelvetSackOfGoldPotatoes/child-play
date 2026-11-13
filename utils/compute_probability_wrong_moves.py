"""Report illegal-reply rates from the archived board-game messages.

Each game ends on its first illegal reply, so the invalid-game share is
well-defined. A per-move probability or mean episode length requires actual
move counts; this script does not estimate them from placeholder arrays.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from plot_fresh_outcome_lines import OLD_CONDITIONS, OLD_DIR, OLD_MODELS, _trial_outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='Optional destination CSV')
    args = parser.parse_args()
    rows = []
    for game in ('tictactoe', 'connectfour', 'battleship'):
        for model in OLD_MODELS:
            for condition in OLD_CONDITIONS:
                folder = OLD_DIR / f'experiment_{game}_{model}_oneshot_{condition}'
                messages = json.loads((folder / f'game_messages_{game}.json').read_text())
                flags = [_trial_outcome(item) for item in messages]
                illegal = sum(item[3] for item in flags)
                rows.append({
                    'Game': game, 'Model': model,
                    'Temperature': float(condition.split('_')[1]),
                    'Archived Games': len(flags), 'Illegal Games': illegal,
                    'Illegal Game Fraction': illegal / len(flags) if flags else None,
                    'Complete 100-game Archive': len(flags) == 100,
                })
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    if args.out:
        result.to_csv(args.out, index=False)


if __name__ == '__main__':
    main()
