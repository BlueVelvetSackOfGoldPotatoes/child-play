from main import TextPlayer, RandomPlayer, DynamicPlayer

class TwoPlayerGame:
    def __init__(self, SpecificGameClass, ask, max_invalid_attempts=2):
        def generateInternalAsk(name):
            async def internalAsk(prompt):        
                player = self._player1 if name == "P1" else self._player2

                messages = []
                messages.append(self._game_instance.prompt)
                messages.append(self._game_instance.get_text_state(player.player_id))
                messages.append(f"Turn {self.turn + 1}") # +1 because turn is counted from 0
                messages.append(f"{player.name}'s turn to guess.")
        
                return await self.ask(messages, prompt)
            
            return internalAsk
        
        def generateChoosePlayer(name):
            is_player1 = name == "P1"
            
            def choosePlayer():
                # If there are still random turns left, we use randomPlayer, otherwise normal player
                if self.random_turns_left > 0:
                    self.random_turns_left -= 1
                    
                    if is_player1: return self._player1_random
                    else: return self._player2_random
                else:
                    if is_player1: return self._player1_normal
                    else: return self._player2_normal

            return choosePlayer
        
        self.ask = ask
        self.max_invalid_attempts = max_invalid_attempts
        
        self.random_turns_left = 0
        
        self._game_instance = SpecificGameClass()

        self._player1_normal = TextPlayer(0, generateInternalAsk("P1"), "P1")
        self._player2_normal = RandomPlayer(1, "P2")

        self._player1_random = RandomPlayer(0, "P1")
        self._player2_random = RandomPlayer(1, "P2")
        
        self._player1 = DynamicPlayer(0, generateChoosePlayer("P1"), "P1")
        self._player2 = DynamicPlayer(1, generateChoosePlayer("P2"), "P2")

        self._game_instance.reset_board()
        
        self._players = [self._player1, self._player2]
        self._current_player_index = 0 if self._game_instance.current_player == "P1" else 1
        self._total_invalid_attempts = [0, 0]
        self._invalid_attempts = [0, 0]  # Track invalid attempts for both players
        self.turn = 0
    
    async def random_moves(self, amount_of_turns=4):
        self.random_turns_left = amount_of_turns
        await self.make_turns(amount_of_turns)
        
        if self.finished:
            self.random_turns_left = 0
            return
        
        if self.random_turns_left > 0:
            self.random_turns_left = 0
            raise Exception("Not all random moves have been played")
        
    async def random_move(self):
        return await self.random_moves(1)
    
    async def make_turns_until_finish(self):
        while not self.finished:
            await self.random_move()
    
    async def make_turns(self, amount_of_turns=1):
        for i in range(amount_of_turns):
            await self.make_turn()

            if self.finished:
                break
    
    async def make_turn(self, extra_messages=[]):
        current_player = self._players[self._current_player_index]

        async def handle_invalid_move(message=None):
            self._invalid_attempts[self._current_player_index] += 1
            self._total_invalid_attempts[self._current_player_index] += 1
            
            if self._invalid_attempts[self._current_player_index] >= self.max_invalid_attempts:
                # End game if max invalid attempts are exceeded
                self._game_instance.game_over = True
                # todo: use winning_message
                winning_message = f"{self._players[1 - self._current_player_index].name} wins by default due to {current_player.name}'s repeated invalid moves."
                # todo: don't return
                return
            
            # the player can still try again, because invalid attempts is less than max invalid attempts
            if message is None:
                message = "Invalid guess"
            return await self.make_turn([*extra_messages, message])

        # todo: use extra messages
        guess = await current_player.make_guess(self._game_instance, previous_play=None) # previous_play is not used

        if guess is None:
            return await handle_invalid_move()

        message, valid_move = self._game_instance.guess(self._current_player_index, guess, current_player)
        
        if not valid_move:
            return await handle_invalid_move(message)

        self._invalid_attempts[self._current_player_index] = 0
            
        if self._game_instance.game_over:
            # todo
            return
        
        self.turn += 1
        self._current_player_index = 0 if self._current_player_index == 1 else 1
    
    async def make_turns_until_finish(self):
        while not self.finished:
            await self.make_turn()
    
    @property
    def finished(self):
        return self._game_instance.game_over
    
    @property
    def invalid_attempts(self):
        return self._total_invalid_attempts[0]