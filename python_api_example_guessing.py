from python_api import GuessingGame
from scripts_games.shapes import Shapes
from scripts_games.countingShapes import CountingShapes

def main():
    game = GuessingGame(CountingShapes)

    for message in game.messages:
        print(message)
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
    print("answer", game.answer)

if __name__ == "__main__":
    main()
