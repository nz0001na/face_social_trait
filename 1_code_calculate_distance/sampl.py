import math

def euclidean_distance(A, B):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(A, B)]))
    return distance


A = (7,	9)
B = (3,	5)
d = euclidean_distance(A, B)
print d

print math.sqrt(32)

