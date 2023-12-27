import copy
from enum import Enum
from queue import Queue
import time
from typing import Optional

DEPTH=3
MAX_STACK_SIZE=8

class FieldColor(Enum):
    BLACK = "Black"
    WHITE = "White"

field_black=FieldColor.BLACK
field_white=FieldColor.WHITE

class CheckerColor(Enum):
    X="X"
    O="O"

checker_black=CheckerColor.X
checker_white=CheckerColor.O

class Field():
    def __init__(self,field_type,checker):
        self.field_type=field_type
        self.stack=[]
        if checker is not None:
            self.stack.append(checker)

    def add_checker(self,checker)->Optional[CheckerColor]:
        length=len(self.stack)
        if length ==0:
            self.stack=checker
        elif length+len(checker)==MAX_STACK_SIZE:
            self.stack=[]
            return True,checker
        else:
            self.stack+=checker
        return False,None

    def is_empty(self)->bool:
        if len(self.stack)>0:
            return False
        return True

    def __str__(self):
        character='.' if self.field_type==field_black else ' '
        matrix = [[character for _ in range(3)] for _ in range(3)]
        for i, checker in enumerate(self.stack):
            if i < MAX_STACK_SIZE:  
                row, col = divmod(i, 3)
                matrix[row][col] = checker.value
        return "\n".join(''.join(row) for row in matrix) + "\n"


class Board():
    def __init__(self, num_of_fields):
        self.num_of_fields = num_of_fields
        self.fields = []

        for row in range(num_of_fields):
            row_fields = []
            for col in range(num_of_fields):
                is_black_field = (row + col) % 2 == 0
                field = Field(field_black if is_black_field else field_white, None)

                if is_black_field and 0 < row < num_of_fields - 1:
                    checker = checker_black if row % 2 == 1 else checker_white
                    field.stack.append(checker)

                row_fields.append(field)
            self.fields.append(row_fields)

    def empty(self):
        for row in self.fields:
            for field in row:
                if not field.is_empty():
                    return False
        return True

    def __str__(self):
        column_labels = "  " + "   ".join(str(i + 1) for i in range(self.num_of_fields))
        board_str = column_labels + "\n"

        for i, row in enumerate(self.fields):
            row_label = chr(65 + i)  
            row_str_lines = ['' for _ in range(3)] 
            for field in row:
                field_lines = field.__str__().split('\n')
                for j in range(3):
                    row_str_lines[j] += field_lines[j] + " "

            board_str += f"{row_label} " + "\n  ".join(row_str_lines) + "\n"

        return board_str

class Player():
    def __init__(self,checker_color):
        self.checker_color=checker_color
        self.score=0

    def add_score(self, points):
        self.score += points

    def __str__(self):
        return f"{self.checker_color.value} - Score: {self.score}"

class Game():
    def __init__(self,board,current_player,player1,player2):
        self.board=board
        self.current_player=current_player
        self.player1=player1
        self.player2=player2
        self.winner=None
        
    def __init__(self):
        self.board=None
        self.winner=None
        self.move_history = []
        self.cache = {}
        self.curr_moves=[]


    def cache_key(self):
        move_history=tuple(self.move_history)
        return (
            self.current_player.checker_color,
            self.player1.score,
            self.player2.score,
            move_history,
        )

    def start(self):
        self.get_board_size()
        player1=Player(checker_white)
        player2=Player(checker_black)
        if(1==self.get_first_player()):
            self.init(player1,player2)
        else:
            self.init(player2,player1)
        self.is_valid_count=0

    def init(self,player1,player2):
        self.board=Board(self.board_size)
        self.current_player=player1
        self.player1=player1
        self.player2=player2

    def get_board_size(self):
        self.board_size = int(input("Board size: "))

    def get_first_player(self):
        print("Choose which goes first: ")
        print("1.O")
        print("2.X")
        return int(input())

    def input_move(self):
        row = input("Enter row (A, B, C, ...): ").upper()
        col = int(input("Enter column (1, 2, 3, ...): "))
        stack_pos = int(input("Enter stack position (0, 1, 2, ...): "))
        direction = input("Enter direction (GL, GD, DL, DD): ").upper()
        return row, col, stack_pos, direction
    
    def bounds_check_and_get_field(self, row, col):
        length=self.board.num_of_fields
        if 0 <= row < length and 0 <= col < length:
            return self.board.fields[row][col]
        else:
            return None 
    
    def bfs(self, start_row, start_col, directions):
        queue = Queue()
        visited = set()
        paths = []

        queue.put([(start_row, start_col, 0)])

        while not queue.empty():
            path = queue.get()
            curr_row, curr_col, distance = path[-1]

            if not self.board.fields[curr_row][curr_col].is_empty() and distance > 0:
                paths.append(path)
                continue 

            if (curr_row, curr_col) not in visited:
                visited.add((curr_row, curr_col))

                for _, (dr, dc) in directions.items():
                    new_row, new_col = curr_row + dr, curr_col + dc

                    if 0 <= new_row < self.board.num_of_fields and 0 <= new_col < self.board.num_of_fields:
                        queue.put(path + [(new_row, new_col, distance + 1)])

        valid_paths = [path for path in paths if not self.board.fields[path[-1][0]][path[-1][1]].is_empty()]

        return valid_paths 
    
    def are_all_directions_empty(self, row_index, col_index, directions):
        for _, (delta_row, delta_col) in directions.items():
            target_row, target_col = row_index + delta_row, col_index + delta_col
            
            move_field=self.bounds_check_and_get_field(target_row,target_col)

            if move_field is not None and not move_field.is_empty():
                return True  
        return False  

    def is_valid_move(self, start_row, start_col, stack_pos, direction):
        self.is_valid_count+=1
        row_index = ord(start_row) - ord('A')
        col_index = start_col - 1

        possible_moves={"GL": (-1, -1), "GD": (-1, 1), "DL": (1, -1), "DD": (1, 1)}
        if direction not in possible_moves:
            return False, "Invalid direction."

        if not (0 <= row_index < self.board.num_of_fields and 0 <= col_index < self.board.num_of_fields):
            return False, "Move is outside the board boundaries."

        if (row_index + col_index) % 2 != 0:
            return False, "Can only move on dark squares."

        start_field = self.board.fields[row_index][col_index]
        if len(start_field.stack) == 0:
            return False, "No stack to move."

        if start_field.stack[stack_pos] != self.current_player.checker_color:
            return False, "You do not own the checker you want to move."

        delta_row, delta_col = possible_moves[direction]
        target_row_index = row_index + delta_row
        target_col_index = col_index + delta_col
        target_field = self.bounds_check_and_get_field(target_row_index,target_col_index)

        if not target_field:
            return False, "Out of bounds"

        if not (0 <= target_row_index < self.board.num_of_fields and 0 <= target_col_index < self.board.num_of_fields):
            return False, "Target position is outside the board boundaries."

        if (target_row_index + target_col_index) % 2 != 0:
            return False, "Can only move to dark squares."

        if len(target_field.stack) > 0:  
            if len(start_field.stack) - stack_pos + len(target_field.stack) >= 9:
                return False, "Cannot form a stack of nine or more."
            
        if not target_field.is_empty():
            target_pos=len(target_field.stack)
            if target_pos < stack_pos or (target_pos == stack_pos and target_pos != 0):
                return False, "Invalid move: The moving checker must be on top of the stack." #promeni komentar

            num_checkers_to_move = len(start_field.stack[stack_pos:])

            resulting_stack_size = len(target_field.stack) + num_checkers_to_move

            if resulting_stack_size <= MAX_STACK_SIZE:
                return True, "Valid move."
            else:
                return False, "Cannot move the stack as it exceeds the maximum size."
        else:
            if self.are_all_directions_empty(row_index,col_index,possible_moves):
                return False, "There is a non-empty, adjacent field."
                
            all_paths=self.bfs(row_index,col_index,possible_moves)

            shortest_paths=sorted(all_paths,key=lambda x:len(x))
            
            shortest_path=shortest_paths[0]

            paths = [path for path in shortest_paths if len(path) == len(shortest_path)]

            move_direction=possible_moves[direction]
            
            for path in paths:
                new_row,new_col,_=path[1]
                test_row=row_index+move_direction[0]
                test_col=col_index+move_direction[1]
                if (test_row,test_col)==(new_row,new_col):
                    return True, "Valid move."

            return False, "There is no valid move."

    def won(self):
        num_of_checkers=((self.board_size-2)*self.board_size/2)
        max_score=num_of_checkers/MAX_STACK_SIZE
        win_score=(2*max_score)//3

        if(self.player1.score>win_score):
            self.winner=self.player1
            return True

        elif(self.player2.score>win_score):
            self.winner=self.player2
            return True

        return False
    
    def is_over(self):
        if self.board.empty():
            return False
        if not self.won():
            return False
        return True

    def switch_player(self):
        self.current_player=self.player1 if self.current_player == self.player2 else self.player2

    def get_move(self):
        print(f"{self.current_player}'s turn:")
        return self.input_move()
    
    def calculate_target_position(self, row_index, col_index, direction):
        dir_offsets = {"GL": (-1, -1), "GD": (-1, 1), "DL": (1, -1), "DD": (1, 1)}
        delta_row, delta_col = dir_offsets[direction]
        return row_index + delta_row, col_index + delta_col
    
    def add_point(self,last_checker):
        if self.current_player.checker_color==last_checker:
            print(last_checker)
            self.current_player.score+=1
            return
        other=self.get_other_player()
        other.score+=1 

    def move(self, row, col, stack_pos, direction):
        row_index = ord(row) - ord('A')
        col_index = col - 1
        start_field = self.board.fields[row_index][col_index]

        stack_to_move = start_field.stack[stack_pos:]
        start_field.stack = start_field.stack[:stack_pos]

        target_row_index, target_col_index = self.calculate_target_position(row_index, col_index, direction)

        target_field = self.board.fields[target_row_index][target_col_index]

        exceeded,last_checker=target_field.add_checker(stack_to_move)
        if exceeded:
            return True, last_checker
        return False,None

    def execute_move(self,move):
        row, col, stack_pos, direction = move
        valid_move,message= self.is_valid_move(row, col, stack_pos, direction)
        if valid_move:
            self.move_history.append(move)
            exceeded, last_checker=self.move(row, col, stack_pos, direction)
            if exceeded:
                self.add_point(last_checker)
        else:
            print(message)
        return valid_move
    
    def generate_moves_from_field(self, row, col):
        moves_from_field = []

        field = self.bounds_check_and_get_field(row, col)
        if not field or field.is_empty():
            return moves_from_field

        for stack_pos in range(len(field.stack)):
            if field.stack[stack_pos] == self.current_player.checker_color:
                for direction in ["GL", "GD", "DL", "DD"]:
                    valid_move, _ = self.is_valid_move(chr(65 + row), col + 1, stack_pos, direction)
                    if valid_move:
                        moves_from_field.append((chr(65 + row), col + 1, stack_pos, direction))

        return moves_from_field

    def generate_all_moves(self,color=None):
        all_moves = []
        if not color:
            color=self.current_player.checker_color

        for row in range(self.board.num_of_fields):
            for col in range(self.board.num_of_fields):
                if self.board.fields[row][col].field_type==field_black:
                    field=self.bounds_check_and_get_field(row,col)
                    if not field:
                       continue 
                    if not field.is_empty() and color in field.stack:
                        moves_from_field = self.generate_moves_from_field(row, col)
                        all_moves.extend(moves_from_field)
        return all_moves
    
    def copy_game(game):
        return copy.deepcopy(game)
    
    def min_max_alpha_beta(self, depth, maximizing_player, alpha, beta):
        key = self.cache_key()
        if key in self.cache:
            return self.cache[key]

        if depth == 0 or self.is_over():
            return self.evaluate_state(), None

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            moves = self.generate_all_moves()
            self.curr_moves=moves

            for move in moves:
                new_game = self.copy_game()
                new_game.execute_move(move)
                eval, _ = new_game.min_max_alpha_beta(depth - 1, False, alpha, beta)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break
            result = max_eval, best_move
            self.cache[key] = result
            return result 
        else:
            min_eval = float('inf')
            best_move = None
            moves = self.generate_all_moves()

            for move in moves:
                new_game = self.copy_game()
                new_game.execute_move(move)
                eval, _ = new_game.min_max_alpha_beta(depth - 1, True, alpha, beta)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break
            result = min_eval, best_move
            self.cache[key] = result
            return result 
        
    def make_ai_move(self):
        depth = DEPTH
        best_score = float('-inf')
        best_move = None

        moves = self.generate_all_moves()

        for move in moves:
            new_game = self.copy_game() 
            new_game.execute_move(move)
            score, _ = new_game.min_max_alpha_beta(depth - 1, False, float('-inf'), float('inf'))

            if score > best_score:
                best_score = score
                best_move = move

        print(best_move)
        self.execute_move(best_move)

    def get_other_player(self):
        if self.current_player==self.player1:
            return self.player2
        return self.player1
    
    def get_num_stacks(self,color):
        num=0
        for row in range(self.board.num_of_fields):
            for col in range(self.board.num_of_fields):
                field = self.board.fields[row][col]
                if field.field_type == field_black and not field.is_empty():
                    bottom_piece_color = field.stack[0]
                    if bottom_piece_color == color:
                        num+=1
        return num
    
    def get_counts_stack(self):
        color=self.current_player.checker_color
        stack_count = 0
        match = 0
        not_match=0
        curr_stacks=0
        other_stacks=0

        for row in range(self.board.num_of_fields):
            for col in range(self.board.num_of_fields):
                field = self.board.fields[row][col]
                
                if field.field_type == field_black and not field.is_empty() and len(field.stack) > 1:
                    stack_count += 1
                    for checker in field.stack:
                        if checker==color:
                            match+=1
                        else:
                            not_match+=1
                        if field.stack[0] == color:
                            curr_stacks+=1
                        else:
                            other_stacks+=1


        return stack_count,match,not_match, curr_stacks, other_stacks

    def calculate_degree_of_stack_control(self,arg):
        total_stacks, color_count, not_color_count = arg 

        if total_stacks == 0:
            return 0.0, 0.0  

        stack_control = color_count / total_stacks
        not_stack_control = not_color_count / total_stacks

        return stack_control,not_stack_control 

    def evaluate_state(self):
        score=0
        curr=self.current_player
        other=self.get_other_player()
        #broj mogucih poteza
        c_count=len(self.curr_moves)
        p_count=len(self.generate_all_moves(other.checker_color))
        if c_count>p_count:
            score+=1
        elif p_count<p_count:
            score-=1
        #ko poseduje vise stackova(prvi element stack-a ai ili player)
        total,c_checkers,p_checkers,c_stacks,p_stacks=self.get_counts_stack()
        if c_stacks>p_stacks:
            score+=1
        elif c_stacks<p_stacks:
            score-=1
        #ko poseduje veci stepen kontrole u stackovima tj. ko u tim stackovima ima ukupno veci broj figura
        c_degree, p_degree = self.calculate_degree_of_stack_control((total,c_checkers,p_checkers))
        if c_degree>p_degree:
            score+=1
        elif c_degree<p_degree:
            score-=1
        #ko ima vise poena kada se napravi potez terminator ili covek
        c_score=curr.score
        p_score=other.score
        if c_score>p_score:
            score+=1
        elif c_score<p_score:
            score-=1

        return score
    
    def __str__(self):
        return f"{self.board}\n" \
               f"Player 1: {self.player1}\n" \
               f"Player 2: {self.player2}\n\n" 

def main():
    game = Game()
    game.start()

    while not game.is_over():
        print(game)
        move=game.get_move()
        game.execute_move(move)
        game.switch_player()
    print(game)
    if game.won():
        print(f"{game.winner} has won!")

def play_vs_ai():
    game = Game()
    game.start()

    while not game.is_over():
        print(game)

        # Potez igrača
        valid=False
        while not valid:
            move=game.get_move()
            valid=game.execute_move(move)
        game.switch_player()

        print(game)


        if not game.is_over():
            game.is_valid_count=0
            start=time.time()
            game.make_ai_move()
            end=time.time()
            print(f"{game.is_valid_count} moves explored in {end-start} seconds")
            game.switch_player()


    print(game)

    if game.won():
        print(f"{game.winner} has won!")

def start_game():
    print("1. Player vs AI")
    print("2. Player vs Player")
    flag=input()
    if(flag=="1"):
        play_vs_ai()
    else:
        main()
    
if __name__=="__main__":
    start_game()