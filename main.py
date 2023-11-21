class Field():
    def __init__(self,color,stack=None):
        self.color=color
        self.stack=[stack]
    def __str__(self):
        idx=len(self.stack)
        if idx>0:
            return f"{self.stack[idx-1]}"
class Board():
    def __init__(self,num_of_fields):
        self.num_of_fields=num_of_fields
        self.fields=[[Field("black","black" if row%2==0 else "white") if (row+col)%2==1 else Field("white"," ") \
                                                                        for col in range(num_of_fields)]    \
                                                                        for row in range (num_of_fields)]
    def __str__(self):
        board_str = ""
        for row in self.fields:
            board_str += ' '.join(str(field) for field in row) + "\n"
        return board_str
def main():
    board=Board(8)
    print(board)
if __name__=="__main__":
    main()
