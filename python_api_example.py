from python_api import TwoPlayerGame
from scripts_games.tictactoe import TicTacToe
from scripts_games.connectfour import ConnectFour
from scripts_games.battleship import BattleShip

import os
import asyncio

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

async def main():
    async def ask(messages, prompt):
        clear()
        
        for message in messages: print(message)

        if len(game.moves) >= 2:
            last_score = game.moves[len(game.moves) - 2]["score"]
            print(f"last score: {last_score}")
        
        return input(prompt)
    
    game = TwoPlayerGame(ConnectFour, ask)

    # runs the game
    # await game.make_turns(amount_of_turns=1)

    # runs random moves for the amount_of_turns 
    # await game.random_moves(amount_of_turns=3)

    # runs the game until finish
    await game.make_turns_until_finish()
    
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
    # number of invalid attempts by custom player
    
if __name__ == "__main__":
    asyncio.run(main())