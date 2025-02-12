import math
import random
import copy
import numpy as np
import matplotlib.pyplot as plt

jacobi_plots = []
def jacobi_method(matrix, answer, n, max_iterations, precision):
    n = len(matrix)
    x = [0 for i in range(n)]
    x_new = [0 for _ in range(n)]
    true_solution = np.linalg.solve(matrix, answer)
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        for i in range(n):
            sigma = sum(matrix[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (answer[i] - sigma) / matrix[i][i]
            
        error = sum((x_new[i] - true_solution[i])**2 for i in range(n))
        jacobi_plots.append((iteration, error))
        
        if error < precision:
            return x_new, iteration
        x = x_new.copy()
    
    return x_new, iteration

gauss_seidel_plots = []
def gauss_seidel_method(matrix, answer, n, max_iterations, precision):
    n = len(matrix)
    x = [0 for i in range(n)]
    x_new = [0 for _ in range(n)]
    true_solution = np.linalg.solve(matrix, answer)
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        for i in range(n):
            sigma_ji = sum(matrix[i][j] * x_new[j] for j in range(i))
            sigma_ij = sum(matrix[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = (answer[i] - sigma_ji - sigma_ij) / matrix[i][i]
            
        error = sum((x_new[i] - true_solution[i])**2 for i in range(n))
        gauss_seidel_plots.append((iteration, error))
        
        if error < precision:
            return x_new, iteration
        x = x_new.copy()
    
    return x_new, iteration

def show_matrix(matrix, answer, upper_bound):
    for i in range(len(matrix)):
        for j in range(len(matrix)+1):
            if j == len(matrix):
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
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        matrix[i][i] = row_sum + random.randint(10, 20)
    return matrix, [random.randint(lower_bound, upper_bound) for _ in range(n)]

jacobi_plots.clear()
gauss_seidel_plots.clear()
n = 20
lower_bound = -100
upper_bound = 200
precision = 1e-10
max_iterations = 100

matrix, answer = generate_diagonal_dominant_matrix_and_answer(n, lower_bound, upper_bound)
print("Original matrix:")
show_matrix(matrix, answer, upper_bound)
print("--------------------------------")
x_jacobi, iterations_jacobi = jacobi_method(matrix, answer, n, max_iterations, precision)
print(f"Jacobi Solution: {x_jacobi} | iterations: {iterations_jacobi}")
x_gauss_seidel, iterations_gauss_seidel = gauss_seidel_method(matrix, answer, n, max_iterations, precision)
print(f"Gauss-Seidel Solution: {x_gauss_seidel} | iterations: {iterations_gauss_seidel}")
print(f"Numpy Solution: {np.linalg.solve(matrix,answer)}")

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(*zip(*jacobi_plots), 'b.-', label='Jacobi')
plt.semilogy(*zip(*gauss_seidel_plots), 'r.-', label='Gauss-Seidel')
plt.grid(True)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title(f'Convergence Speed Jacobi vs Gauss-Seidel \n(matrices size={n}x{n}, value range={(lower_bound, upper_bound)}, precision={precision})')
plt.legend()
plt.show()