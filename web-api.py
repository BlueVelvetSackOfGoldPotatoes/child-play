from main import play_one_game, TextPlayer
from scripts_games.battleship import BattleShip
from scripts_games.connectfour import ConnectFour
from scripts_games.tictactoe import TicTacToe

from cuid2 import cuid_wrapper
import asyncio

cuid = cuid_wrapper()

class GameWrapper:
    def __init__(self, Game, callback=None):
        self.callback = callback
        
        self.game_messages = []
        
        self.awaiting_guess = None
        self.guess_event = asyncio.Event()
        self.guess_value = None

        def ask(name):
            async def inner(prompt):
                player = self.player1 if name == "P1" else self.player2

                messages = []
                messages.append(self.game_instance.prompt)
                messages.append(self.game_instance.get_text_state(player.player_id))
                messages.append(f"{player.name}'s turn to guess.")

                self.awaiting_guess = (player.player_id, messages, prompt)
                if self.callback is not None:
                    self.callback(player.player_id, messages, prompt)

                await self.guess_event.wait()
                self.guess_event.clear()
                return self.guess_value
            
            return inner

        self.game_instance = Game()
        self.player1 = TextPlayer(0, ask("P1"), "P1")
        self.player2 = TextPlayer(1, ask("P2"), "P2")
        
    def collect_game_message(self, message):
        self.game_messages.append(message)
        
    async def start(self):
        await play_one_game(self.game_instance, self.player1, self.player2, None, message_callback=self.collect_game_message)
        
    def guess(self, guess):
        if self.awaiting_guess is None:
            raise ValueError("No guess is currently awaited.")
        
        self.awaiting_guess = None
        self.guess_value = guess
        self.guess_event.set()

from fastapi import FastAPI

app = FastAPI()

game_classes = {
    "tictactoe": TicTacToe,
    "battleship": BattleShip,
    "connectfour": ConnectFour
}
games = {}

@app.post("/games/empty")
async def create_empty_game(game_name: str):
    if game_name not in game_classes:
        raise ValueError(f"Unknown game: {game_name}")
    
    GameClass = game_classes[game_name]
    
    game = GameWrapper(GameClass)
    id = cuid()
    games[id] = game
    
    asyncio.create_task(game.start())
    
    return {"id": id}

@app.get("/games/{id}/messages")
async def get_game(id: str):
    if id not in games:
        raise ValueError(f"Unknown game: {id}")
    
    return games[id].game_messages

@app.get("/games/{id}/current")
async def get_game_current(id: str):
    if id not in games:
        raise ValueError(f"Unknown game: {id}")
    
    (player_id, messages, prompt) = games[id].awaiting_guess
    return {
        "player_id": player_id,
        "messages": messages,
        "prompt": prompt
    }

@app.post("/games/{id}")
async def make_guess(id: str, guess: str):
    if id not in games:
        raise ValueError(f"Unknown game: {id}")
    
    games[id].guess(guess)