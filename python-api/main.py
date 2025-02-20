from main import play_one_game, TextPlayer

class Game:
    def __init__(self, SpecificGameClass):
        def generateAsk(name):
            async def ask(prompt):        
                player = player1 if name == "P1" else player2

                messages = []
                messages.append(game_instance.prompt)
                messages.append(game_instance.get_text_state(player.player_id))
                messages.append(f"{player.name}'s turn to guess.")
        
                pass
            
            return ask
        
        game_instance = SpecificGameClass()
        player1 = TextPlayer(0, generateAsk("P1"), "P1")
        player2 = TextPlayer(1, generateAsk("P2"), "P2")

        max_invalid_attempts = 2 # todo: make customizable

        game_instance.reset_board()
        
        players = [player1, player2]
        current_player_index = 0 if game_instance.current_player == "P1" else 1
        total_invalid_attempts = [0, 0]
        invalid_attempts = [0, 0]  # Track invalid attempts for both players
        wrong_moves = [0, 0] #todo: what is this? and make api for this # Track wrong moves (both invalid and incorrect) for both players
        move_log = []
        turn = 0

        previous_play = ""
        
        while not game_instance.game_over:
            current_player = players[current_player_index]

            guess = await current_player.make_guess(game_instance, previous_play=None) # previous_play is not used
            previous_play = guess

            if guess is None:
                invalid_attempts[current_player_index] += 1
                total_invalid_attempts[current_player_index] += 1
                wrong_moves[current_player_index] += 1
                if invalid_attempts[current_player_index] >= max_invalid_attempts:
                    # End game if max invalid attempts are exceeded
                    game_instance.game_over = True
                    # todo: use winning_message
                    winning_message = f"{players[1 - current_player_index].name} wins by default due to {current_player.name}'s repeated invalid moves."
                    # todo: don't return
                    return

            message, valid_move = game_instance.guess(current_player_index, guess, current_player)
            # todo: use message
            
            if not valid_move:
                # todo: combine with same code above
                invalid_attempts[current_player_index] += 1
                total_invalid_attempts[current_player_index] += 1
                wrong_moves[current_player_index] += 1
                if invalid_attempts[current_player_index] >= max_invalid_attempts:
                    # End game if max invalid attempts are exceeded
                    game_instance.game_over = True
                    # todo: use winning_message
                    winning_message = f"{players[1 - current_player_index].name} wins by default due to {current_player.name}'s repeated invalid moves."
                    # todo: don't return
                    return
                
                # todo: what to do now???
            else:
                invalid_attempts[current_player_index] = 0
                
            if game_instance.game_over:
                # todo
                return
            
            turn += 1
            current_player_index = 0 if current_player_index == 1 else 1

        raise Exception("This shouldn't be run")
    
    async def random_moves(amount_of_turns=4):
        pass
    
    async def run(amount_of_turns=None):
        pass