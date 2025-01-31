from main import play_one_game, TextPlayer
from scripts_games.battleship import BattleShip
from scripts_games.connectfour import ConnectFour
from scripts_games.tictactoe import TicTacToe
import os

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

game_messages = []
def collect_game_message(message):
    game_messages.append(message)

def ask(name):
    def inner(prompt):
        player = player1 if name == "P1" else player2

        messages = []
        messages.append(game_instance.prompt)
        messages.append(game_instance.get_text_state(player.player_id))
        messages.append(f"{player.name}'s turn to guess.")

        clear()
        for message in messages: print(message)
        return input(prompt)
    
    return inner

game_instance = TicTacToe()
player1 = TextPlayer(0, ask("P1"), "P1")
player2 = TextPlayer(1, ask("P2"), "P2")

play_one_game(game_instance, player1, player2, None, message_callback=collect_game_message)

clear()
for message in game_messages: print(message)