from python_api import GuessingGame
from scripts_games.shapes import Shapes

import asyncio

async def main():
    game = GuessingGame(Shapes)

    for message in game.messages:
        print(message)
    print(game.text_state)

    guess = input(game.prompt)

    valid, correct, score, answer = await game.guess(guess)

    print("valid", valid)
    print("correct", correct)
    print("score", score)
    print("answer", answer)

if __name__ == "__main__":
    asyncio.run(main())
