from main import play_one_game, TextPlayer

class Game:
    def __init__(self, SpecificGameClass):
        def generateAsk(name):
            async def ask(prompt):        
                player = self._player1 if name == "P1" else self._player2

                messages = []
                messages.append(self._game_instance.prompt)
                messages.append(self._game_instance.get_text_state(player.player_id))
                messages.append(f"{player.name}'s turn to guess.")
        
                pass
            
            return ask
        
        self._game_instance = SpecificGameClass()
        self._player1 = TextPlayer(0, generateAsk("P1"), "P1")
        self._player2 = TextPlayer(1, generateAsk("P2"), "P2")

        self.max_invalid_attempts = 2 # todo: make customizable

        self._game_instance.reset_board()
        
        self._players = [self._player1, self._player2]
        self._current_player_index = 0 if self._game_instance.current_player == "P1" else 1
        self._total_invalid_attempts = [0, 0]
        self._invalid_attempts = [0, 0]  # Track invalid attempts for both players
        self._wrong_moves = [0, 0] #todo: what is this? and make api for this # Track wrong moves (both invalid and incorrect) for both players
        self.turn = 0
    
    async def random_moves(amount_of_turns=4):
        pass
    
    async def random_move():
        pass
    
    async def make_turns(self, amount_of_turns=1):
        for i in range(amount_of_turns):
            await self.make_turn()

            # todo: use game.finished  
            if self._game_instance.game_over:
                break
    
    async def make_turn(self):
        current_player = self._players[self._current_player_index]

        guess = await current_player.make_guess(self._game_instance, previous_play=None) # previous_play is not used

        if guess is None:
            self._invalid_attempts[self._current_player_index] += 1
            self._total_invalid_attempts[self._current_player_index] += 1
            self._wrong_moves[self._current_player_index] += 1
            if self._invalid_attempts[self._current_player_index] >= self.max_invalid_attempts:
                # End game if max invalid attempts are exceeded
                self._game_instance.game_over = True
                # todo: use winning_message
                winning_message = f"{self._players[1 - self._current_player_index].name} wins by default due to {current_player.name}'s repeated invalid moves."
                # todo: don't return
                return

        message, valid_move = self._game_instance.guess(self._current_player_index, guess, current_player)
        # todo: use message
        
        if not valid_move:
            # todo: combine with same code above
            self._invalid_attempts[self._current_player_index] += 1
            self._total_invalid_attempts[self._current_player_index] += 1
            self._wrong_moves[self._current_player_index] += 1
            if self._invalid_attempts[self._current_player_index] >= self.max_invalid_attempts:
                # End game if max invalid attempts are exceeded
                self._game_instance.game_over = True
                # todo: use winning_message
                winning_message = f"{self._players[1 - self._current_player_index].name} wins by default due to {current_player.name}'s repeated invalid moves."
                # todo: don't return
                return
            
            # todo: what to do now???
        else:
            self._invalid_attempts[self._current_player_index] = 0
            
        if self._game_instance.game_over:
            # todo
            return
        
        self.turn += 1
        self._current_player_index = 0 if self._current_player_index == 1 else 1
    
    async def make_turns_until_finish(self):
        # todo: use self.finished
        while not self._game_instance.game_over:
            await self.make_turn()