import math
import random
import copy
import numpy as np

def jacobi_method(matrix, 
                  answer, 
                  n, 
                  max_iterations, 
                  precision, 
                  printing=False,
                  print_precision=5):
    n = len(matrix)
    x = [random.uniform(-100, 100) for i in range(n)]
    x_new = [0 for _ in range(n)]
    for iterations in range(max_iterations):
        if printing:
            print(f"Iteration {iterations+1}: ~{list(map(lambda x: round(x, print_precision), x))}")
        for i in range(n):
            sigma = sum(matrix[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (answer[i] - sigma) / matrix[i][i]
        if should_stop(x, x_new, precision):
            break
        x = copy.deepcopy(x_new)
    
    return x_new, iterations+1

def should_stop(v0, v1, epsilon):
    s = 0
    for i in range(len(v0)):
        s += (v0[i] - v1[i]) ** 2
    return epsilon > s
        
def show_matrix(matrix, answer, upper_bound):
    for i in range(n):
        for j in range(n+1):
            if j == n:
                print(f" | {answer[i]}")
            else:
                print_str = str(matrix[i][j])
                while len(print_str) < len(str(upper_bound*18)):
                    print_str = print_str + " "
                print(print_str, end=" ")
            
        print()

def generate_diagonal_dominant_matrix_and_answer(n, lower_bound, upper_bound):
    matrix = [[random.randint(lower_bound, upper_bound) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        matrix[i][i] = random.randint(upper_bound*9, upper_bound*18)
    answer = [random.randint(lower_bound, upper_bound) for _ in range(n)]
    return matrix, answer

n = 8
matrix, answer = generate_diagonal_dominant_matrix_and_answer(n, lower_bound=-1000, upper_bound=1000)
print("Original matrix:")
show_matrix(matrix, answer, upper_bound=10000)
x, iterations = jacobi_method(matrix, answer, n, max_iterations=100, precision=1e-12, printing=True, print_precision=5)
print(f"Jacobi Solution: {x} | iterations: {iterations}")
print(f"Numpy Solution: {np.linalg.solve(matrix,answer)}")