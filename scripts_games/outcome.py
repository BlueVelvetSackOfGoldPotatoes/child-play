"""Classify one ChildPlay move without loading the model clients.

The same classifier is used by the published alternating-play runner and by
the local Qwen runs, so an illegal reply is an immediate loss in both.
"""


def apply_move(game, player, move):
    """Apply one move and classify the game's response.

    Args:
        game: Tic-tac-toe, Connect Four, or Battleship instance.
        player: 0 for the model, 1 for the random opponent.
        move: A column, or a row-column pair, as the game expects.

    Returns:
        str: One of continue, win, tie, invalid.
    """
    result = game.guess(player, move, None)
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(f"{game.name}.guess must return a (message, valid) pair")
    message, ok = result
    if not ok:
        return "invalid"
    if message == "Win":
        return "win"
    if message == "Tie":
        return "tie"
    return "continue"
