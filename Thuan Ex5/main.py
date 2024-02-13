import cmath
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, exp

TOLERANCE = 1e-5
ITERATION = 200000000

def answer():
        
    a = 2.1601
    b = 3.7439
    c = 5.6771
    d = -2.5837
    r0 = -2
    j0 = 3
    h = 0.001

def newton_raphson(g, g_derivative, initial_guess, r0, j0, min_abs, min_value, max_iterations=1000, tolerance=1.3):
    # output is an estimation of the root of f 
    # using the Newton Raphson method
    # iterative implementation

    for _ in range(max_iterations):
        val = g(initial_guess, r0, j0)

        if abs(val) < min_abs:
            min_abs = abs(val)
            min_value = val

        if abs(val) < tolerance:
            return val
        else:
            initial_guess = initial_guess - g(initial_guess, r0, j0) / g_derivative(initial_guess, r0, j0)

    return min_value

def implicit_euler(r_derivative_equation, j_deriavative_equation, r_2derivative_equation, j_2derivative_equation, guess_r1, guess_j1, r0, j0, h):

    def rg(initial_guess,r,j):
        return initial_guess - r0 - h * r_derivative_equation(initial_guess, r, j)

    def jg(initial_guess,r,j):
        return initial_guess - j0 - h * j_deriavative_equation(initial_guess, r, j)

    def rg_derivative(initial_guess,r,j):
        return 1 - h * r_2derivative_equation(initial_guess, r, j)
    
    def jg_derivative(initial_guess,r,j):
        return 1 - h * j_2derivative_equation(initial_guess, r, j)

    
    r1 = newton_raphson(rg,rg_derivative,guess_r1, r0, j0,float('inf'), float('inf') , ITERATION, h**2)
    j1 = newton_raphson(jg,jg_derivative,guess_j1, r0, j0,float('inf'), float('inf') , ITERATION, h**2)
    return r1, j1


def ExplicitEuler(r_derivative_equation, j_deriavative_equation, t0, R0, J0, h):
    R1 = R0 + r_derivative_equation(t0, R0, J0) * h
    J1 = J0 + j_deriavative_equation(t0, R0, J0) * h
    return R1, J1

def solve_quadratic(a, b, c):
    # calculate the discriminant
    d = (b**2) - (4*a*c)

    # find two solutions
    sol1 = (-b-cmath.sqrt(d))/(2*a)
    sol2 = (-b+cmath.sqrt(d))/(2*a)

    if d > 0:
        return sol1, sol2, 2
    elif d == 0:
        return sol1, sol2, 1
    else:
        return sol1, sol2, -2


def solve_RJ(a, b, c, d):
    rho = a + d
    sigma = a*d - b*c

    lambda1, lambda2, n = solve_quadratic(1,-rho, sigma)

    if n == 2:
        
        def r_equation_two_distinct_real_roots(t, r0, j0):
            c1 = (j0*b + r0*(a-lambda2))/(b*(lambda2 - lambda1))
            c2 =(-j0*b - r0*(a-lambda1))/(b*(lambda2 - lambda1))
            return c1 * (-b) * cmath.e ** (lambda1 * t) + c2 * (-b) * cmath.e ** (lambda2 * t)
        
        def r_derivative_equation_two_distinct_real_roots(t, r0, j0):
            c1 = (j0*b + r0*(a-lambda2))/(b*(lambda2 - lambda1))
            c2 =(-j0*b - r0*(a-lambda1))/(b*(lambda2 - lambda1))
            return c1 * (-b) * lambda1 * cmath.e ** (lambda1 * t) + c2 * (-b) * lambda2 * cmath.e ** (lambda2 * t)

        def r_2derivative_equation_two_distinct_real_roots(t, r0, j0):
            c1 = (j0*b + r0*(a-lambda2))/(b*(lambda2 - lambda1))
            c2 =(-j0*b - r0*(a-lambda1))/(b*(lambda2 - lambda1))
            return c1 * (-b) * lambda1**2 * cmath.e ** (lambda1 * t) + c2 * (-b) * lambda2 ** 2 * cmath.e ** (lambda2 * t)

        def j_equation_two_distinct_real_roots(t, r0, j0):
            c1 = (j0*b + r0*(a-lambda2))/(b*(lambda2 - lambda1))
            c2 =(-j0*b - r0*(a-lambda1))/(b*(lambda2 - lambda1))
            return c1 * (a - lambda1) * cmath.e ** (lambda1 * t) + c2 * (a - lambda2) * cmath.e ** (lambda2 * t)
        
        def j_derivative_equation_two_distinct_real_roots(t, r0, j0):
            c1 = (j0*b + r0*(a-lambda2))/(b*(lambda2 - lambda1))
            c2 =(-j0*b - r0*(a-lambda1))/(b*(lambda2 - lambda1))
            return c1 * (a - lambda1) * lambda1 * cmath.e ** (lambda1 * t) + c2 * (a - lambda2) * lambda2 * cmath.e ** (lambda2 * t)

        def j_2derivative_equation_two_distinct_real_roots(t, r0, j0):
            c1 = (j0*b + r0*(a-lambda2))/(b*(lambda2 - lambda1))
            c2 =(-j0*b - r0*(a-lambda1))/(b*(lambda2 - lambda1))
            return c1 * (a - lambda1) * lambda1 **2 * cmath.e ** (lambda1 * t) + c2 * (a - lambda2) * lambda2**2 * cmath.e ** (lambda2 * t)

        return r_equation_two_distinct_real_roots, j_equation_two_distinct_real_roots, r_derivative_equation_two_distinct_real_roots, j_derivative_equation_two_distinct_real_roots, r_2derivative_equation_two_distinct_real_roots, j_2derivative_equation_two_distinct_real_roots
        
    elif n == -2:
        yamma = a - d
        delta = cmath.sqrt(abs((a-d)**2 + 4*b*c))
        
        
        def r_equation_two_conjugate_complex_roots(t, r0, j0):
            c1 = j0 / (2*c)
            c2 = (r0 - c1 * yamma)/delta
    
            new_t = delta * t / 2
            #eq1 = yamma * cmath.cos(new_t) - delta * cmath.sin(new_t) 
            #eq2 = delta * cmath.cos(new_t) + yamma * cmath.sin(new_t)
            
            eq1 = (yamma * c1 + delta * c2)
            eq2 = (-delta * c1 + yamma * c2)
            eq3 =  cmath.cos(new_t)
            eq4 =  cmath.sin(new_t) 

            #return (eq1 * c1 + eq2 * c2) * cmath.e ** ((rho * t)/2)
            return (eq3 * eq1 + eq4 * eq2) * cmath.e ** ((rho * t)/2)
        

        def r_derivative_equation_two_conjugate_complex_roots(t, r0, j0):
            c1 = j0 / (2*c)
            c2 = (r0 - c1 * yamma)/delta
    
            new_t = delta * t / 2
            half_delta = delta / 2
            half_rho = rho / 2
            
            eq1 = (yamma * c1 + delta * c2)
            eq2 = (-delta * c1 + yamma * c2)
            eq3 =  cmath.cos(new_t)
            eq4 =  cmath.sin(new_t) 
            
            return (eq4 * (half_rho * eq2 - eq1 * half_delta) + eq3 * (half_rho * eq1 + half_delta * eq2) ) * cmath.e ** (half_rho * t)
    
        def j_equation_two_conjugate_complex_roots(t, r0, j0):
            c1 = j0 / (2*c)
            c2 = (r0 - c1 * yamma)/delta
    
            new_t = delta * t / 2
            eq1 = 2*c * cmath.cos(new_t)
            eq2 = 2*c * cmath.sin(new_t) 
            return (eq1 * c1 + eq2 * c2) * cmath.e ** ((rho * t)/2)

        
        def j_derivative_equation_two_conjugate_complex_roots(t, r0, j0):
            c1 = j0 / (2*c)
            c2 = (r0 - c1 * yamma)/delta
            
            new_t = delta * t / 2
            half_delta = delta / 2
            half_rho = rho / 2

            eq1 = 2*c
            eq3 =  cmath.cos(new_t)
            eq4 =  cmath.sin(new_t) 
            return (eq3 * (half_rho + half_delta) + eq4 * (half_rho - half_delta)) * eq1 * cmath.e ** ((rho * t)/2)

        return r_equation_two_conjugate_complex_roots, j_equation_two_conjugate_complex_roots, r_derivative_equation_two_conjugate_complex_roots, j_derivative_equation_two_conjugate_complex_roots
    else :
        a_matrix = np.array([[a, b], [c, d]])
        identity_matrix = np.identity(2)
        zero_matrix = np.zeros((2, 2))

        # Subtract lambda1 * Identity Matrix from the original matrix
        result = a_matrix - lambda1 * identity_matrix

        # Check if the result is the zero matrix
        is_zero_matrix = np.array_equal(result, zero_matrix)

        if is_zero_matrix:
            def r_equation_independent_eigenvectors(t, r0, j0):
                return r0 * cmath.e ** (lambda1 * t)
            
            def r_derivative_equation_independent_eigenvectors(t, r0, j0):
                return r0 * lambda1 * cmath.e ** (lambda1 * t)

            def j_equation_independent_eigenvectors(t, r0, j0):
                return (j0) * cmath.e ** (lambda1 * t)
            
            def j_derivative_equation_independent_eigenvectors(t, r0, j0):
                return (j0) * lambda1 * cmath.e ** (lambda1 * t)

            return r_equation_independent_eigenvectors, j_equation_independent_eigenvectors, r_derivative_equation_independent_eigenvectors, j_derivative_equation_independent_eigenvectors

        else :
            # Solve the system of linear equations
            def r_equation_dependent_eigenvectors(t, r0, j0):    
                c1 = -r0 / b
                c2 = -j0 + (-r0 * (a-lambda1))/b
                e1 =  cmath.e ** (lambda1 * t)
                return (c1 * (-b)) * e1 + c2 * ((-b) * t * e1)
            
            def r_derivative_equation_dependent_eigenvectors(t, r0, j0):
                c1 = -r0 / b
                c2 = -j0 + (-r0 * (a-lambda1))/b
                e1 =  cmath.e ** (lambda1 * t)
                return (c1 * (-b)) * lambda1 * e1 + c2 * ((-b) * e1) *  (lambda1 * t+1)

            def j_equation_dependent_eigenvectors(t, r0, j0):
                c1 = -r0 / b
                c2 = -j0 + (-r0 * (a-lambda1))/b
                e1 =  cmath.e ** (lambda1 * t)
                return (c1 * (a - lambda1)) * e1 + c2 * (((a-lambda1) * t * e1) - e1)
            
            def j_derivative_equation_dependent_eigenvectors(t, r0, j0):
                c1 = -r0 / b
                c2 = -j0 + (-r0 * (a-lambda1))/b
                e1 =  cmath.e ** (lambda1 * t)
                return (c1 * (a - lambda1)) * lambda1 * e1 + c2 * e1 * ( (a-lambda1) * lambda1 * t + (a-lambda1) - lambda1) # Here I am not sure about the derivative
            
            return r_equation_dependent_eigenvectors, j_equation_dependent_eigenvectors, r_derivative_equation_dependent_eigenvectors, j_derivative_equation_dependent_eigenvectors
    

    return None, None, None, None


def show_data():
    
    data = pd.read_csv('exact.csv')

    # Check the first few rows of the data
    print(data.head())
    
    # Plot the data (R in red and J in blue)
    plt.scatter(data.index, data['R'], color='red', s = 0.5)
    plt.scatter(data.index, data['J'], color='blue', s = 0.5)



    # Show the plot
    plt.show()


def main():
    r_equation, j_equation, r_derivative_equation, j_derivative_equation, r_2derivative_equation, j_2derivative_equation = solve_RJ(2.1601, 3.7439, 5.6771, -2.5837)

    
    data = pd.read_csv('exact.csv')

    
    # Plot the data (R in red and J in blue)
    plt.scatter(data.index, data['R'], color='red', s = 0.5)
    plt.scatter(data.index, data['J'], color='blue', s = 0.5)

    current_r , current_j = (-2 ,3)
    plt.scatter(0, current_r, color='green', s = 0.5)
    plt.scatter(0, current_j, color='black', s = 0.5)

    k = 1
    for t in range(1, 1000 * k):
        #current_r , current_j = ExplicitEuler(r_derivative_equation,j_derivative_equation,0.001 ,current_r,current_j, 0.001/k)
        #print(f"t: {t/k}")

        current_r , current_j = implicit_euler(r_derivative_equation,j_derivative_equation, r_2derivative_equation, j_2derivative_equation, current_r, current_r, current_r,current_j, 0.001/k)
        
        plt.scatter(t/k, current_r, color='green', s = 0.5)
        plt.scatter(t/k, current_j, color='black', s = 0.5)

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()

