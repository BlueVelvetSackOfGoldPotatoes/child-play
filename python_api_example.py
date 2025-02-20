from python_api import TwoPlayerGame
from scripts_games.tictactoe import TicTacToe
from scripts_games.connectfour import ConnectFour
from scripts_games.battleship import BattleShip

import os
import asyncio

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

async def ask(messages, prompt):
    clear()
    for message in messages: print(message)
    return input(prompt)

async def main():
    # Game instance
    game = TwoPlayerGame(ConnectFour, ask)

    # runs the game
    await game.make_turns(amount_of_turns=1)

    # runs random moves for the amount_of_turns 
    await game.random_moves(amount_of_turns=3)

    # runs the game until finish
    await game.make_turns_until_finish()
    
    clear()
    
    # print("moves", game.moves)
    # [
    #     {
    #         player: "custom", "opponent" or "random"
    #         guess: ... # differs per game
    #         score: 0-1
    #     }
    # ]
    
    print("winner", game.winner)
    # None (if game not finished), "tie", "custom", "opponent" or "random"
    
    print("invalid attempts", game.invalid_attempts)
    # number of invalid attempts by custom player
    
    print("finished", game.finished)
    # boolean if game is finished
    
if __name__ == "__main__":
    asyncio.run(main())