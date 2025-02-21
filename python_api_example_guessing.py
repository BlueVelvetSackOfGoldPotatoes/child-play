from python_api import GuessingGame, LCLValidity, LCLGenerateConstruct
from scripts_games.shapes import Shapes
from scripts_games.countingShapes import CountingShapes

import os

def clear():
    os.system("cls" if os.name == "nt" else "clear")

def main():
    game = GuessingGame(LCLGenerateConstruct)

    clear()
    for message in game.messages:
        print(message)
    if game.text_state: # depends on the game
        print(game.text_state)

    valid = False
    while not valid:
        guess = input(game.prompt)

        valid, correct, score, message = game.guess(guess)

        if not valid:
            print(message)

    print("")
    print("valid", valid)
    print("correct", correct)
    print("score", score)
    print("message", message)
    if game.answer: # depends on the game
        print("answer", game.answer)

if __name__ == "__main__":
    main()
