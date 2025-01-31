from main import play_one_game, TextPlayer
from scripts_games.battleship import BattleShip
from scripts_games.connectfour import ConnectFour
from scripts_games.tictactoe import TicTacToe

class GameWrapper:
    def __init__(self, Game, callback):
        self.game_messages = []

        def ask(name):
            def inner(prompt):
                player = self.player1 if name == "P1" else self.player2

                messages = []
                messages.append(self.game_instance.prompt)
                messages.append(self.game_instance.get_text_state(player.player_id))
                messages.append(f"{player.name}'s turn to guess.")

                return callback(player.player_id, messages, prompt)
            
            return inner

        self.game_instance = Game()
        self.player1 = TextPlayer(0, ask("P1"), "P1")
        self.player2 = TextPlayer(1, ask("P2"), "P2")
        
    def collect_game_message(self, message):
        self.game_messages.append(message)
        
    def start(self):
        play_one_game(self.game_instance, self.player1, self.player2, None, message_callback=self.collect_game_message)

# from fastapi import FastAPI

# app = FastAPI()

