from main import TextPlayer, RandomPlayer, DynamicPlayer
from lcl import LCLGame
import random

class TwoPlayerGame:
    def __init__(self, SpecificGameClass, ask, max_invalid_attempts=2):
        def generateInternalAsk(name):
            async def internalAsk(prompt, extra_messages=[]):
                player = self._player1 if name == "P1" else self._player2

                messages = []
                messages.append(self._game_instance.prompt)
                messages.append(self._game_instance.get_text_state(player.player_id))
                messages.append(
                    f"Turn {self.turn + 1}"
                )  # +1 because turn is counted from 0
                messages.append(f"{player.name}'s turn to guess.")

                for message in extra_messages:
                    messages.append(message)

                return await self.ask(messages, prompt)

            return internalAsk

        def generateChoosePlayer(name):
            is_player1 = name == "P1"

            def choosePlayer():
                # If there are still random turns left, we use randomPlayer, otherwise normal player
                if self.random_turns_left > 0:
                    self.random_turns_left -= 1

                    if is_player1:
                        return self._player1_random
                    else:
                        return self._player2_random
                else:
                    if is_player1:
                        return self._player1_normal
                    else:
                        return self._player2_normal

            return choosePlayer

        self.ask = ask
        self.max_invalid_attempts = max_invalid_attempts

        self.random_turns_left = 0
        self.winner = (
            None  # None (if game not finished), "tie", "custom", "opponent" or "random"
        )

        self._game_instance = SpecificGameClass()

        self._player1_normal = TextPlayer(0, generateInternalAsk("P1"), "P1")
        self._player2_normal = RandomPlayer(1, "P2")

        self._player1_random = RandomPlayer(0, "P1")
        self._player2_random = RandomPlayer(1, "P2")

        self._player1 = DynamicPlayer(0, generateChoosePlayer("P1"), "P1")
        self._player2 = DynamicPlayer(1, generateChoosePlayer("P2"), "P2")

        self._game_instance.reset_board()

        self._players = [self._player1, self._player2]
        self._current_player_index = (
            0 if self._game_instance.current_player == "P1" else 1
        )
        self._total_invalid_attempts = [0, 0]
        self._invalid_attempts = [0, 0]  # Track invalid attempts for both players
        self.turn = 0
        self.moves = []

    @property
    def finished(self):
        return self._game_instance.game_over

    @property
    def invalid_attempts(self):
        return self._total_invalid_attempts[0]

    @property
    def text_state(self):
        return self._game_instance.get_text_state(self._players[0].player_id)

    async def random_moves(self, amount_of_turns=4):
        if self.finished:
            return

        self.random_turns_left = amount_of_turns
        await self.make_turns(amount_of_turns)

        if self.finished:
            self.random_turns_left = 0
            return

        if self.random_turns_left > 0:
            self.random_turns_left = 0
            raise Exception("Not all random moves have been played")

    async def random_move(self):
        if self.finished:
            return

        return await self.random_moves(1)

    async def random_moves_until_finish(self):
        if self.finished:
            return

        while not self.finished:
            await self.random_move()

    async def make_turns(self, amount_of_turns=1):
        if self.finished:
            return

        for i in range(amount_of_turns):
            await self.make_turn()

            if self.finished:
                break

    async def make_turns_until_finish(self):
        if self.finished:
            return

        while not self.finished:
            await self.make_turn()

    async def make_turn(self, extra_messages=[]):
        if self.finished:
            return

        current_player = self._players[self._current_player_index]
        is_random_player = self.random_turns_left > 0

        async def handle_invalid_move(message=None):
            self._invalid_attempts[self._current_player_index] += 1
            self._total_invalid_attempts[self._current_player_index] += 1

            if (
                self._invalid_attempts[self._current_player_index]
                >= self.max_invalid_attempts
            ):
                # End game if max invalid attempts are exceeded
                self._game_instance.game_over = True

                # other player wins
                if self._current_player_index == 0:
                    self.winner = "opponent"
                else:
                    self.winner = "custom"

                return

            # the player can still try again, because invalid attempts is less than max invalid attempts
            return await self.make_turn([*extra_messages, message])

        guess = await current_player.make_guess(
            self._game_instance, previous_play=None, extra_messages=extra_messages
        )  # previous_play is not used

        if guess is None:
            return await handle_invalid_move("Invalid guess")

        message, valid_move, score = self._game_instance.guess(
            self._current_player_index, guess, current_player
        )

        if not valid_move:
            return await handle_invalid_move(f"Guess: {guess}\n{message}")

        self._invalid_attempts[self._current_player_index] = 0

        if message == "Win" or message == "Tie" or message == "Loss":
            self._game_instance.game_over = True

        if self._game_instance.game_over:
            if message == "Tie":
                self.winner = "tie"
            elif message == "Win":
                if is_random_player:
                    self.winner = "random"
                elif self._current_player_index == 0:
                    self.winner = "custom"
                else:
                    self.winner = "opponent"
            elif message == "Loss":
                if self._current_player_index == 0:
                    self.winner = "opponent"
                else:
                    self.winner = "custom"
            else:
                raise Exception(f"Unknown game.guess return message: {message}")

        if is_random_player:
            move_player_name = "random"
        elif self._current_player_index == 0:
            move_player_name = "custom"
        else:
            move_player_name = "opponent"

        self.moves.append(
            {
                "player": move_player_name,
                "player_index": self._current_player_index,
                "guess": guess,
                "score": score,
            }
        )

        if self._game_instance.game_over:
            return

        self.turn += 1
        self._current_player_index = 1 - self._current_player_index

class GuessingGame:
    def __init__(self, GameClass):
        if GameClass.__name__ == "Shapes":
            self._game = "shapes"
        elif GameClass.__name__ == "CountingShapes":
            self._game = "countingShapes"
        elif GameClass.__name__ == "LCLValidity":
            self._game = "lclValidity"
        else:
            self._game = "general"
        
        # Game instance and answer
        if self._game == "shapes":
            self.answer = random.choice(GameClass.possible_shapes)
            self._game_instance = GameClass(shape=self.answer)
        elif self._game == "countingShapes":
            self._game_instance = GameClass()
            
            total_shapes_amount = random.randrange(1, 10)
            possible_shape_symbols = list(GameClass.possible_shapes.values())
            possible_color_symbols = list(GameClass.possible_colors.values())

            shapes = [
                (random.choice(possible_shape_symbols), random.choice(possible_color_symbols))
                for _ in range(total_shapes_amount)
            ]
            
            for shape in shapes:
                self._game_instance.place_shapes(shape[0], shape[1], 1)
        
            # could be different then total_shapes_amount, because of overlapping shapes, or the game could stop placing new shapes, because board is full
            self.answer = self._game_instance.count_shapes()
        else:
            self._game_instance = GameClass()
            self.answer = self._game_instance.answer

        # Setting all the correct text attributes
        self.messages = []
        
        # Game prompt
        self.messages.append(self._game_instance.prompt)
        
        # Other messages
        if self._game == "shapes":
            text_answers = "Answers:\n" + "\n".join([f"{i}: {option}" for i, option in enumerate(self._game_instance.answer_options)])
            self.messages.append(text_answers)
        
        # Text state
        if self._game == "shapes":
            self.text_state = "\n".join("".join(row) for row in self._game_instance.board)
        elif self._game == "countingShapes":
            self.text_state = "\n".join(" ".join(row) for row in self._game_instance.board)
        else:
            self.text_state = self._game_instance.get_text_state()
        
        # Prompt
        self.prompt = "Enter your guess: "

    def guess(self, guess):
        # Parse guess
        if self._game == "shapes" or self._game == "countingShapes":
            try:
                guess = int(guess)
            except ValueError:
                valid = False
                correct = False
                score = 0.0
                message = "Invalid guess. Guess is not an integer."
                
                return valid, correct, score, message
        elif self._game == "lclValidity":
            if guess == "valid" or guess == "invalid":
                guess = guess == "valid"
            else:
                valid = False
                correct = False
                score = 0.0
                message = "Invalid guess. Guess is not valid or invalid."
                
                return valid, correct, score, message
        
        if self._game == "countingShapes":
            # guess is always valid if it's an integer
            valid = True
            correct = guess == self.answer

            message = None
            if not correct:
                message = self._game_instance.compare_count(guess)
        elif self._game == "lclValidity":
            # guess is always valid if it's a boolean
            valid = True
            
            message = None
            correct = self._game_instance.guess(guess)
        else:
            message, valid = self._game_instance.guess(guess)

            if valid:
                if message == "Win":
                    correct = True
                else:
                    correct = False
            else:
                correct = False

            if valid:
                message = None

        if not valid:
            score = 0.0
        elif correct:
            score = 1.0
        elif self._game == "countingShapes":
            difference = abs(self.answer - guess)
            score = 1 / (pow(difference, 2) + 1)
        else:
            score = 0.0
        
        return valid, correct, score, message

class LCLValidity:
    def __init__(self):
        self._game_instance = LCLGame()

        self.prompt = (
            f"You will receive a description of a Lego structure, for instance, [(x1, y1, 'color1'), "
            f"(x2, y2, 'color2')], which lists the coordinates and colors of two pieces. A construct is "
            f"valid if all Lego pieces are connected but not overlapping. A Lego piece is connected through "
            f"interlocking pegs, not by merely touching sides. Two Lego pieces overlap when they share the "
            f"same y-coordinate and any part of their length has the same x-coordinate. If the following "
            f"structure is valid then reply with valid, otherwise reply with invalid (do not justify your "
            f"answer)"
        )
        
        self.reset()

    def reset(self):
        self._game_instance.pieces = []
        self._pieces = []
        
        self.answer = random.choice([True, False]) # Whether it's a valid move
        self._pieces = self._game_instance.generate_valid_or_invalid_construct(5, valid=self.answer)

    def get_text_state(self):
        return f"{self._pieces}"

    def guess(self, guess):
        return guess == self.answer
