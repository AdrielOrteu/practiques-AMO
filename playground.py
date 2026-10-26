import numpy as np
a = (1,2,3)
m = np.array([[1,0,0],[0,0,1],[0,1,0]])
a = np.array(a)
a = m@a
a = tuple(map(int, a))
print(a)
tmp = {}
orientation = {(1,0,0):"r", (0,0,-1):"b"}
print()
for vect, color in orientation.items():
    v = np.array(vect)
    v = m @ v
    v = tuple(map(int, v))
    print(v)
    print(vect)
    tmp[v] = color
    print()
orientation = tmp
print(orientation)