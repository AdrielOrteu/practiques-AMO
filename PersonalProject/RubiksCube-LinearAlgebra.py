import numpy as np
import itertools


class Cubelet:
    def __init__(self, og_position: tuple):
        colors = {
            (1, 0, 0): "r",
            (0, 1, 0): "g",
            (0, 0, 1): "w",
            (-1, 0, 0): "o",
            (0, -1, 0): "b",
            (0, 0, -1): "y"
        }
        self.orientation_m = np.diag(og_position)
        self.orientation: dict[tuple, str] = {
            tuple(int(val) for val in col): colors[tuple(int(val) for val in col)]
            for col in self.orientation_m
            if tuple(int(val) for val in col) in colors
        }
    
    def change_orientation(self, matrix):
        self.orientation_m = matrix @ self.orientation_m
        tmp = {}
        for vect, color in self.orientation.items():
            v = np.array(vect, dtype=float)
            v = matrix @ v
            # Use round and int to avoid floating-point truncation bugs
            v = tuple(int(round(val)) for val in v)
            tmp[v] = color
        self.orientation = tmp


class RubiksCube:
    def __init__(self):
        self.c_space: dict[tuple, Cubelet] = {}
    
    def init_solved_cube(self):
        cublet_pos_vectors = list(itertools.product([-1, 0, 1], repeat=3))
        for cublet_vector in cublet_pos_vectors:
            if cublet_vector == (0, 0, 0):
                continue
            self.c_space[cublet_vector] = Cubelet(cublet_vector)
    
    def rotate(self, matrix, face):
        tmp_c_space = {}
        for vect, cubelet in self.c_space.items():
            # Check if cubelet lies on the requested slice using dot product
            if sum(face[i] * vect[i] for i in range(3)) > 0:
                v = np.array(vect, dtype=float)
                v = matrix @ v
                v = tuple(int(round(val)) for val in v)
                cubelet.change_orientation(matrix)
                tmp_c_space[v] = cubelet
            else:
                tmp_c_space[vect] = cubelet
        self.c_space = tmp_c_space
    
    def show(self):
        COLORS = {
            'o': '\033[38;5;208m',  # Orange
            'y': '\033[93m',  # Yellow
            'g': '\033[92m',  # Green
            'b': '\033[94m',  # Blue
            'w': '\033[97m',  # White
            'r': '\033[91m',  # Red
            'RESET': '\033[0m'
        }
        
        def get_color_char(char):
            return f"{COLORS.get(char, '')}{char}{COLORS['RESET']}"
        
        indent = " " * 9
        
        # 1. Top face (z = 1)
        top_face = []
        for y in [1, 0, -1]:
            row = []
            for x in [-1, 0, 1]:
                cubelet = self.c_space.get((x, y, 1))
                char = cubelet.orientation.get((0, 0, 1), ' ') if cubelet else ' '
                row.append(char)
            top_face.append(row)
        
        # 2. Bottom face (z = -1)
        bottom_face = []
        for y in [-1, 0, 1]:
            row = []
            for x in [-1, 0, 1]:
                cubelet = self.c_space.get((x, y, -1))
                char = cubelet.orientation.get((0, 0, -1), ' ') if cubelet else ' '
                row.append(char)
            bottom_face.append(row)
        
        # 3. Middle rows: Left (x=-1), Front (y=1), Right (x=1), Back (y=-1)
        middle_rows = []
        for z in [1, 0, -1]:
            left_row = [
                self.c_space[(-1, y, z)].orientation.get((-1, 0, 0), ' ')
                if (-1, y, z) in self.c_space else ' '
                for y in [1, 0, -1]
            ]
            front_row = [
                self.c_space[(x, 1, z)].orientation.get((0, 1, 0), ' ')
                if (x, 1, z) in self.c_space else ' '
                for x in [-1, 0, 1]
            ]
            right_row = [
                self.c_space[(1, y, z)].orientation.get((1, 0, 0), ' ')
                if (1, y, z) in self.c_space else ' '
                for y in [-1, 0, 1]
            ]
            back_row = [
                self.c_space[(x, -1, z)].orientation.get((0, -1, 0), ' ')
                if (x, -1, z) in self.c_space else ' '
                for x in [1, 0, -1]
            ]
            middle_rows.append((left_row, front_row, right_row, back_row))
        
        # Print Layout
        for row in top_face:
            print(indent + " ".join(get_color_char(c) for c in row))
        
        for left, front, right, back in middle_rows:
            l_str = " ".join(get_color_char(c) for c in left)
            f_str = " ".join(get_color_char(c) for c in front)
            r_str = " ".join(get_color_char(c) for c in right)
            b_str = " ".join(get_color_char(c) for c in back)
            print(f"{l_str}   {f_str}   {r_str}   {b_str}")
        
        for row in bottom_face:
            print(indent + " ".join(get_color_char(c) for c in row))


# Test execution:
cube = RubiksCube()
cube.init_solved_cube()
cube.show()

keep_going = True
while keep_going:
    R = [[float(input(f"Row {i}, Col {j}: ")) for j in range(3)] for i in range(3)]
    R = np.array(R)
    print(R)
    f = input("Write the vector you want to rotate around [format -> x y z]: ")
    f = tuple(int(x) for x in f.split())
    cube.rotate(R, f)
    cube.show()
    
    keep_going = "y" == input("Do you want to keep going (y/n)? ").strip().lower()