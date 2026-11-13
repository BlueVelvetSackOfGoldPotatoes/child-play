"""Count immediate tactical errors on actual accepted moves, by episode."""

import numpy as np
import re


def episodes_from_messages(messages, rows, cols):
    """Recover accepted moves from the actual before/after board observations.

    Each message-list is one retained game, including games with no accepted
    move. Duplicate final-board observations do not create duplicate moves.
    """
    episodes = []
    symbols = {'': 0, '.': 0, 'X': 1, 'O': 2}
    for game_messages in messages:
        moves = []
        previous = None
        for message in game_messages:
            text = re.sub(r'\x1b\[[0-9;]*m', '', str(message))
            grid = []
            for line in text.splitlines():
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(cells) == cols and all(cell in symbols for cell in cells):
                    grid.append([symbols[cell] for cell in cells])
            if len(grid) != rows:
                continue
            board = np.asarray(grid)
            if previous is None:
                if board.any():
                    raise ValueError('archive is missing the initial empty board')
            else:
                changed = np.argwhere(board != previous)
                if len(changed) == 1:
                    row, col = map(int, changed[0])
                    if previous[row, col] != 0 or board[row, col] == 0:
                        raise ValueError('archive overwrites a played cell')
                    move = col if rows == 7 else [row, col]
                    moves.append({'player': int(board[row, col]) - 1, 'move': move, 'turn': len(moves)})
                elif len(changed) > 1:
                    raise ValueError('archive skips a board transition')
            previous = board
        if previous is None:
            raise ValueError('archive has no readable board observation')
        episodes.append(moves)
    return episodes


def episodes_from_moves(logs):
    """Accept explicit episodes, or split legacy move records at turn zero."""
    if not logs:
        return []
    if isinstance(logs[0], list):
        return logs
    episodes = []
    current = []
    for move in logs:
        if current and move.get('turn') == 0:
            episodes.append(current)
            current = []
        current.append(move)
    if current:
        episodes.append(current)
    return episodes


def _can_prevent_threat(board, own, opponent, gravity, is_winning_move, can_win_next):
    """Require a legal way to avoid the opponent's immediate win."""
    if gravity:
        candidates = []
        for col in range(board.shape[1]):
            empty = np.flatnonzero(board[:, col] == 0)
            if len(empty):
                candidates.append((int(empty[-1]), col))
    else:
        candidates = np.argwhere(board == 0)
    for row, col in candidates:
        board[row, col] = own
        prevented = is_winning_move(board, own) or not can_win_next(board, opponent)
        board[row, col] = 0
        if prevented:
            return True
    return False


def count_moves(logs, rows, cols, gravity, is_winning_move, can_win_next):
    """Count a missed win only if it existed before a non-winning move.

    A missed block requires an opponent win before the move that remains
    available afterward despite a legal way to prevent it. Winning immediately
    takes priority over blocking; unavoidable double threats are excluded.
    Empty episodes are retained when the caller provides explicit episodes.
    Invalid or discontinuous accepted-move records raise rather than silently
    producing a plausible tactical score.
    """
    episodes = episodes_from_moves(logs)
    total_moves = {0: 0, 1: 0}
    missed_wins = {0: 0, 1: 0}
    missed_blocks = {0: 0, 1: 0}
    for moves in episodes:
        board = np.zeros((rows, cols), dtype=int)
        for turn, entry in enumerate(moves):
            player = entry['player']
            if player != turn % 2:
                raise ValueError('accepted moves must alternate from player 0')
            own, opponent = player + 1, 2 - player
            win_available = can_win_next(board, own)
            block_needed = can_win_next(board, opponent)
            block_available = block_needed and _can_prevent_threat(
                board, own, opponent, gravity, is_winning_move, can_win_next
            )
            if gravity:
                col = entry['move']
                if not 0 <= col < cols:
                    raise ValueError('out-of-range accepted column')
                empty = np.flatnonzero(board[:, col] == 0)
                if not len(empty):
                    raise ValueError('accepted move into a full column')
                row = int(empty[-1])
            else:
                row, col = entry['move']
                if not (0 <= row < rows and 0 <= col < cols):
                    raise ValueError('out-of-range accepted cell')
            if board[row, col] != 0:
                raise ValueError('accepted move into an occupied cell')
            board[row, col] = own
            total_moves[player] += 1
            won = is_winning_move(board, own)
            if not won:
                missed_wins[player] += int(win_available)
                missed_blocks[player] += int(block_available and can_win_next(board, opponent))
            if (won or not (board == 0).any()) and turn != len(moves) - 1:
                raise ValueError('accepted moves continue after a terminal board')
    n = len(episodes)
    return {
        'average_moves': {p: total_moves[p] / n if n else float('nan') for p in (0, 1)},
        'missed_wins': missed_wins,
        'missed_blocks': missed_blocks,
        'total_games': n,
    }
