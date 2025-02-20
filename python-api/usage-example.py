from .main import Game
from .two_player_games import TicTacToe

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
    game = Game(TicTacToe)

    # runs random moves for the amount_of_turns 
    await game.random_moves(amount_of_turns=2)
    
    # runs the game
    await game.make_turns(amount_of_turns=1)

    # runs the game until finish
    await game.make_turns_until_finish()
    
    print("moves", game.moves)
    # [
    #     {
    #         player: "custom", "opponent" or "random"
    #         guess: ... # differs per game
    #         score: 0-1
    #     }
    # ]
    
    print("winner", game.winner)
    # None (for tie), "custom", "opponent" or "random"
    
    print("invalid moves", game.invalid_moves)
    # number of invalid moves by custom player
    
    print("finished", game.finished)
    # boolean if game is finished
    
if __name__ == "__main__":
    asyncio.run(main())