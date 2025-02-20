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

    await game.make_turns_until_finish()

    clear()
    print(game.moves)
    print(game.text_state)
    # None (if game not finished), "tie", "custom", "opponent" or "random"
    print("winner", game.winner)
    print("invalid attempts", game.invalid_attempts)


if __name__ == "__main__":
    asyncio.run(main())
