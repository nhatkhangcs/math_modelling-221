import cmath
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, diff, exp

TOLERANCE = 1e-5
MAX_ITERATION = 200000000
DERIVATIVE = 1e-12

def answer():
        
    a = 2.1601
    b = 3.7439
    c = 5.6771
    d = -2.5837
    r0 = -2
    j0 = 3
    h = 0.001

def derivative(f):
    def df_dr(t, r, j):
        return (f(t, r + DERIVATIVE, j) - f(t, r, j)) / DERIVATIVE
    
    def df_dj(t, r, j):
        h = 1e-6  # Small perturbation
        return (f(t, r, j + DERIVATIVE) - f(t, r, j)) / DERIVATIVE
    
    return df_dr, df_dj


def newton_raphson(F, J_F, initial_guess):
    # Newton-Raphson iteration
    for _ in range(MAX_ITERATION):
        # Compute function values
        val = F(initial_guess)

        # Check convergence
        if np.linalg.norm(val) < TOLERANCE:
            return initial_guess

        # Compute increments using Newton-Raphson formula
        increments = np.linalg.solve(J_F(initial_guess), val)
        initial_guess -= increments.astype(float) # Update initial guess

    raise ValueError("Newton-Raphson did not converge")

def implicit_euler(f, g,t0, r0, j0, h):
    # Derivatives of f and g
    df_dr, df_dj = derivative(f)
    dg_dr, dg_dj = derivative(g)

    # Function for F(R_{n+1}, J_{n+1})
    def F(R_J):
        R, J = R_J
        return np.array([R - r0 - h * f(t0, R, J), J - j0 - h * g(t0, R, J)])

    # Jacobian matrix of F(R_{n+1}, J_{n+1})
    def J_F(R_J):
        R, J = R_J
        return np.array([[1 - h * df_dr(t0, R, J), -h * df_dj(t0, R, J)],
                         [-h * dg_dr(t0, R, J), 1 - h * dg_dj(t0, R, J)]])

    # Initial guess for Newton-Raphson
    guess = np.array([r0, j0], dtype=float)

    # Newton-Raphson iteration    
    R1, J1 = newton_raphson(F, J_F, guess)

    return R1, J1

def ExplicitEuler(f, g, t0, R0, J0, h):
    R1 = R0 + f(t0, R0, J0) * h
    J1 = J0 + g(t0, R0, J0) * h
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

        #def r_2derivative_equation_two_distinct_real_roots(t, r0, j0):
        #    c1 = (j0*b + r0*(a-lambda2))/(b*(lambda2 - lambda1))
        #    c2 =(-j0*b - r0*(a-lambda1))/(b*(lambda2 - lambda1))
        #    return c1 * (-b) * lambda1**2 * cmath.e ** (lambda1 * t) + c2 * (-b) * lambda2 ** 2 * cmath.e ** (lambda2 * t)

        def j_equation_two_distinct_real_roots(t, r0, j0):
            c1 = (j0*b + r0*(a-lambda2))/(b*(lambda2 - lambda1))
            c2 =(-j0*b - r0*(a-lambda1))/(b*(lambda2 - lambda1))
            return c1 * (a - lambda1) * cmath.e ** (lambda1 * t) + c2 * (a - lambda2) * cmath.e ** (lambda2 * t)
        
        def j_derivative_equation_two_distinct_real_roots(t, r0, j0):
            c1 = (j0*b + r0*(a-lambda2))/(b*(lambda2 - lambda1))
            c2 =(-j0*b - r0*(a-lambda1))/(b*(lambda2 - lambda1))
            return c1 * (a - lambda1) * lambda1 * cmath.e ** (lambda1 * t) + c2 * (a - lambda2) * lambda2 * cmath.e ** (lambda2 * t)

        #def j_2derivative_equation_two_distinct_real_roots(t, r0, j0):
        #    c1 = (j0*b + r0*(a-lambda2))/(b*(lambda2 - lambda1))
        #    c2 =(-j0*b - r0*(a-lambda1))/(b*(lambda2 - lambda1))
        #    return c1 * (a - lambda1) * lambda1 **2 * cmath.e ** (lambda1 * t) + c2 * (a - lambda2) * lambda2**2 * cmath.e ** (lambda2 * t)

        return r_equation_two_distinct_real_roots, j_equation_two_distinct_real_roots, r_derivative_equation_two_distinct_real_roots, j_derivative_equation_two_distinct_real_roots
        
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
    r_equation, j_equation, r_derivative_equation, j_derivative_equation = solve_RJ(2.1601, 3.7439, 5.6771, -2.5837)

    
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

        current_r , current_j = implicit_euler(r_derivative_equation,j_derivative_equation, 0, current_r,current_j, 0.001/k)
        
        plt.scatter(t/k, current_r, color='green', s = 0.5)
        plt.scatter(t/k, current_j, color='black', s = 0.5)

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()

