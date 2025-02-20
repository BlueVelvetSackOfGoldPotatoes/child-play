import copy

class ConnectFour:
    def __init__(self, options=None):
        if options is None:
            self.rows = 7
            self.cols = 7
            options = {}
        else:
            self.debug = options.get("debug", False)
            self.rows = options.get("rows", 7)
            self.cols = options.get("cols", 7)
        
        self.name = "connectfour"
        self.reset_board()
        self.last_move = (-1, -1)
        self.game_over = False
        self.current_player = "P1"  # Assuming P1 starts the game
        self.prompt = "Connect-Four is a two-player game. The pieces fall straight down, occupying the next available space within a column. The objective of the game is to be the first to form a horizontal, vertical, or diagonal line of four of one's own discs. In a board, player 1, you, plays with symbol X, while player 2, your opponent, plays with symbol O. Your input is just a number from 0 to 6, nothing else.  Do not output anything else but the col value else you lose."

    def reset_board(self):
        self.board = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        self.current_player = "P1"
        self.moves_made = []
        self.game_over = False

    def reset_game(self):
        self.board = [["." for _ in range(self.cols)] for _ in range(self.rows)]

    def check_tie(self):
        return all(self.board[0][col] != '.' for col in range(self.cols)) and not self.check_win()

    def check_win(self, new_board = None, last_move = None):
        if new_board is None:
            new_board = self.board
        if last_move is None:
            last_move = self.last_move
        
        row, col = last_move
        if row == -1:
            return False
        player = new_board[row][col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for d in [1, -1]:
                r, c = row, col
                while 0 <= r + d*dr < self.rows and 0 <= c + d*dc < self.cols and new_board[r + d*dr][c + d*dc] == player:
                    count += 1
                    r += d*dr
                    c += d*dc
                    if count >= 4:
                        return True
        return False
    
    def check_loss(self) -> bool:
        return False

    def guess(self, player_index, guess, player):
        col = guess
        if col < 0 or col >= self.cols:
            return False
        for row in reversed(range(self.rows)):
            if self.board[row][col] == ".":

                previous_board = copy.deepcopy(self.board)
                self.board[row][col] = "X" if player_index == 0 else "O"
                self.last_move = (row, col)
                
                if self.check_win():
                    self.game_over = True
                    return "Win", True, 1.0
                if self.check_tie():
                    self.game_over = True
                    return "Tie", True, 0.5
                
                self.switch_player()
                score = self.calculate_score(previous_board, (row, col), player_index)
                return "Valid move", True, score
        return "Invalid move.", False, None
    
    def calculate_score(self, previous_board, guess, player_index):
        """Calculates the score of a move in Connect 4 based on its impact."""
        import copy
        
        row, col = guess
        symbol = "X" if player_index == 0 else "O"
        opponent_symbol = "O" if symbol == "X" else "X"
        score = 0.0
        
        # Clone the previous board
        new_board = copy.deepcopy(previous_board)
        
        new_board[row][col] = symbol  # Apply the move
        
        # Check if the move results in a win
        if self.check_win(new_board, (row, col)):
            return 1.0
        
        directions = [
            [(row + i, col) for i in range(-3, 4) if 0 <= row + i < self.rows],  # Vertical
            [(row, col + i) for i in range(-3, 4) if 0 <= col + i < self.cols],  # Horizontal
            [(row + i, col + i) for i in range(-3, 4) if 0 <= row + i < self.rows and 0 <= col + i < self.cols],  # Diagonal \
            [(row + i, col - i) for i in range(-3, 4) if 0 <= row + i < self.rows and 0 <= col - i < self.cols]  # Diagonal /
        ]
        
        for line in directions:
            symbols = [new_board[r][c] for r, c in line]
            if symbols.count(opponent_symbol) == 3 and symbols.count(".") == 0:
                score = max(score, 0.8)  # Blocking opponent's winning move
            elif symbols.count(opponent_symbol) == 2 and symbols.count(".") == 0:
                score = max(score, 0.7)  # Blocking opponent's 2 in a row
                
            if symbols.count(symbol) == 3 and symbols.count(".") >= 1:
                score = max(score, 0.7)  # Creating three in a row
            elif symbols.count(symbol) == 2 and symbols.count(".") >= 2:
                score = max(score, 0.6)  # Creating unblocked two in a row
            elif symbols.count(symbol) == 2 and symbols.count(".") >= 1:
                score = max(score, 0.3)  # Creating two in a row
        
        if score > 0.9:
            score = 0.9
        
        return score
    
    def get_text_state(self, player_index=None):
        red = "\033[91mX\033[0m"
        yellow = "\033[32mO\033[0m"
        state_lines = [" 0 1 2 3 4 5 6"]
        for row in self.board:
            row_str = "|"
            for cell in row:
                if cell == "X":
                    row_str += red + "|"
                elif cell == "O":
                    row_str += yellow + "|"
                else:
                    row_str += cell + "|"
            state_lines.append(row_str)
        return "\n".join(state_lines)

    def switch_player(self):
        """Switches the current player."""
        self.current_player = "P2" if self.current_player == "P1" else "P1"

    @property
    def board_size(self):
        return self.cols  