from python_api import TwoPlayerGame
from scripts_games.tictactoe import TicTacToe
from scripts_games.connectfour import ConnectFour
from scripts_games.battleship import BattleShip

import os
import asyncio


def clear():
    os.system("cls" if os.name == "nt" else "clear")


async def ask(messages, prompt):
    # Implement your own model here

    clear()
    for message in messages:
        print(message)

    return input(prompt)


async def main():
    game = TwoPlayerGame(ConnectFour, ask)

    await game.make_turns(1)

    await game.random_moves(3)

    if not game.finished:
        await game.make_turns_until_finish()

    # game is finished

    clear()

    print("moves", game.moves)
    # [
    #     {
    #         player: "custom", "opponent" or "random"
    #         player_index: 0 or 1
    #         guess: ... # differs per game
    #         score: 0-1
    #     }
    # ]

    print(game.text_state)

    print("winner", game.winner)
    # None (if game not finished), "tie", "custom", "opponent" or "random"

    print("invalid attempts", game.invalid_attempts)


if __name__ == "__main__":
    asyncio.run(main())
