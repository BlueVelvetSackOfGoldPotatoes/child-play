from main import TextPlayer, RandomPlayer, DynamicPlayer
from lcl import LCLGame

import random
import re
import ast

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

        self._game_instance = SpecificGameClass()

        self._player1_normal = TextPlayer(0, generateInternalAsk("P1"), "P1")
        self._player2_normal = RandomPlayer(1, "P2")

        self._player1_random = RandomPlayer(0, "P1")
        self._player2_random = RandomPlayer(1, "P2")

        self._player1 = DynamicPlayer(0, generateChoosePlayer("P1"), "P1")
        self._player2 = DynamicPlayer(1, generateChoosePlayer("P2"), "P2")

        self._players = [self._player1, self._player2]
        
        self.reset()

    @property
    def finished(self):
        return self._game_instance.game_over

    @property
    def invalid_attempts(self):
        return self._total_invalid_attempts[0]

    @property
    def text_state(self):
        return self._game_instance.get_text_state(self._players[0].player_id)
    
    def reset(self):
        self._game_instance.reset_board()
        
        self.random_turns_left = 0
        self.winner = (
            None  # None (if game not finished), "tie", "custom", "opponent" or "random"
        )
        
        self._current_player_index = (
            0 if self._game_instance.current_player == "P1" else 1
        )
        self._total_invalid_attempts = [0, 0]
        self._invalid_attempts = [0, 0]  # Track invalid attempts for both players
        self.turn = 0
        self.moves = []

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

            new_extra_messages = list(extra_messages)
            if message is not None:
                new_extra_messages.append(message)
                
            return await self.make_turn(new_extra_messages)

        guess = await current_player.make_guess(
            self._game_instance, previous_play=None, extra_messages=extra_messages
        )  # previous_play is not used

        if guess is None:
            return await handle_invalid_move()

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
        elif GameClass.__name__ == "LCLGenerateConstruct":
            self._game = "lclGenerateConstruct"
        else:
            self._game = "general"

        self._GameClass = GameClass
        self._game_instance = None
        self.reset()
        
    def reset(self):
        # Game instance and answer
        if self._game == "shapes":
            self.answer = random.choice(self._GameClass.possible_shapes)
            if self._game_instance is None:
                self._game_instance = self._GameClass(shape=self.answer)
            else:
                self._game_instance.shape = self.answer
                self._game_instance.reset_board()
        elif self._game == "countingShapes":
            if self._game_instance is None:
                self._game_instance = self._GameClass()
            else:
                self._game_instance.reset_board()
            
            total_shapes_amount = random.randrange(1, 10)
            possible_shape_symbols = list(self._GameClass.possible_shapes.values())
            possible_color_symbols = list(self._GameClass.possible_colors.values())

            shapes = [
                (random.choice(possible_shape_symbols), random.choice(possible_color_symbols))
                for _ in range(total_shapes_amount)
            ]
            
            for shape in shapes:
                self._game_instance.place_shapes(shape[0], shape[1], 1)
        
            # could be different then total_shapes_amount, because of overlapping shapes, or the game could stop placing new shapes, because board is full
            self.answer = self._game_instance.count_shapes()
        elif self._game == "lclGenerateConstruct":
            if self._game_instance is None:
                self._game_instance = self._GameClass()
            # lclGenerateConstruct doesn't need a reset, as there's no state
            self.answer = None # lclGenerateConstruct doesn't have one correct answer
        else:
            if self._game_instance is None:
                self._game_instance = self._GameClass()
            else:
                self._game_instance.reset()
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
        elif self._game == "lclGenerateConstruct":
            self.text_state = None # lclGenerateConstruct doesn't have a text state
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
        
        if self._game == "lclGenerateConstruct":
            return self._game_instance.guess(guess)
        
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

class LCLGenerateConstruct:
    def __init__(self):
        self._game_instance = LCLGame()
        self.pieces_amount = 3

        self.prompt = (
            f"A description of a Lego structure consists of a list of tuples, "
            f"[(x1, y1, 'color1'), (x2, y2, 'color2')], where each tuple shows the coordinates "
            f"and colors of a piece. Such a structure is valid if all Lego pieces are connected "
            f"but not overlapping. A Lego piece is connected through interlocking pegs, not by "
            f"merely touching sides. Two Lego pieces overlap when they share the same y-coordinate "
            f"and any part of their length has the same x-coordinate. Produce a description of a valid "
            f"structure using {self.pieces_amount} Lego pieces. Reply only with the Lego structure description "
            f"following the format [(x1, y1, 'color1'), (x2, y2, 'color2'), ...], write nothing else "
            f"but the structure."
        )

    def guess(self, guess):
        if isinstance(guess, str):
            match = re.search(r'\[\s*(.*?)\s*\]', guess, re.DOTALL)

            if not match:
                valid = False
                correct = False
                score = 0.0
                message = "Guess is not in the correct type"
                return valid, correct, score, message

            # Use the captured group from the match, which is the content inside the brackets
            content_to_evaluate = match.group(0)  # Include the brackets

            # Evaluate the extracted string content
            evaluated_response = ast.literal_eval(content_to_evaluate)

            # Check if evaluated_response is empty
            if not evaluated_response:
                valid = False
                correct= False
                score = 0.0
                message = "Guess is an empty list"
                return valid, correct, score, message

            # Flatten if necessary
            if len(evaluated_response) == 1 and isinstance(evaluated_response[0], tuple) and all(isinstance(item, tuple) for item in evaluated_response[0]):
                # It's a list with one element, which is a tuple of tuples
                evaluated_response = list(evaluated_response[0])

            guess = evaluated_response
            
        valid = True
        message = None
        correct = self._game_instance.is_valid_construct(guess)
        
        if correct:
            if len(guess) != self.pieces_amount:
                correct = False
                score = 0.5
                message = "Guess does not have the correct amount of pieces"
                
                return valid, correct, score, message
            
            score = 1.0
        else:
            score = 0.0

        return valid, correct, score, message