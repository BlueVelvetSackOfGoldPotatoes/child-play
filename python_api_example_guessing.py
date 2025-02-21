from python_api import GuessingGame
from scripts_games.shapes import Shapes

def main():
    game = GuessingGame(Shapes)

    for message in game.messages:
        print(message)
    print(game.text_state)

    guess = input(game.prompt)

    valid, correct, score, answer, message = game.guess(guess)

    print("valid", valid)
    print("correct", correct)
    print("score", score)
    print("answer", answer)
    print("message", message)

if __name__ == "__main__":
    main()
