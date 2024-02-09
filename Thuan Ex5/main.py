import cmath
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def answer():
        
    a = 2.1601
    b = 3.7439
    c = 5.6771
    d = -2.5837
    r0 = -2
    j0 = 3
    h = 0.001


def ExplicitEuler(f, g, t0, R0, J0, h):
    R1 = R0 + f(t0, R0, J0) * h
    J1 = J0 + g(t0, R0, J0) * h
    return R1, J1

def ImplicitEuler(f, g, t0, R0, J0, h):
    R1 = R0 + f(t0 + h, R1, J1) * h # R1 J1 is unknown
    J1 = J0 + g(t0 + h, R1, J1) * h
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


def solve_RJ(a, b, c, d, r0, j0):
    rho = a + d
    sigma = a*d - b*c

    lambda1, lambda2, n = solve_quadratic(1,-rho, sigma)

    if n == 2:
        c1 = (j0*b + r0*(a-lambda2))/(b*(lambda2 - lambda1))
        c2 =(-j0*b - r0*(a-lambda1))/(b*(lambda2 - lambda1))
        
        def r_equation_two_distinct_real_roots(t):
            return c1 * (-b) * cmath.e ** (lambda1 * t) + c2 * (-b) * cmath.e ** (lambda2 * t)
        def j_equation_two_distinct_real_roots(t):
            return c1 * (a - lambda1) * cmath.e ** (lambda1 * t) + c2 * (a - lambda2) * cmath.e ** (lambda2 * t)
        
        return r_equation_two_distinct_real_roots, j_equation_two_distinct_real_roots, lambda1, lambda2

    elif n == -2:
        yamma = a - d
        delta = cmath.sqrt(abs((a-d)**2 + 4*b*c))
        c1 = j0 / (2*c)
        c2 = (r0 - c1 * yamma)/delta
        
        def r_equation_two_conjugate_complex_roots(t):
            new_t = delta * t / 2
            eq1 = yamma * cmath.cos(new_t) - delta * cmath.sin(new_t) 
            eq2 = delta * cmath.cos(new_t) + yamma * cmath.sin(new_t)
            return (eq1 * c1 + eq2 * c2) * cmath.e ** ((rho * t)/2)
        def j_equation_two_conjugate_complex_roots(t):
            new_t = delta * t / 2
            eq1 = 2*c * cmath.cos(new_t)
            eq2 = 2*c * cmath.sin(new_t) 
            return (eq1 * c1 + eq2 * c2) * cmath.e ** ((rho * t)/2)

        return r_equation_two_conjugate_complex_roots, j_equation_two_conjugate_complex_roots        

    else :
        a_matrix = np.array([[a, b], [c, d]])
        identity_matrix = np.identity(2)
        zero_matrix = np.zeros((2, 2))

        # Subtract lambda1 * Identity Matrix from the original matrix
        result = a_matrix - lambda1 * identity_matrix

        # Check if the result is the zero matrix
        is_zero_matrix = np.array_equal(result, zero_matrix)

        if is_zero_matrix:
            def r_equation_independent_eigenvectors(t):
                return r0 * cmath.e ** (lambda1 * t)
            
            def j_equation_independent_eigenvectors(t):
                return (j0) * cmath.e ** (lambda1 * t)

            return r_equation_independent_eigenvectors, j_equation_independent_eigenvectors

        else :
            # Solve the system of linear equations
            c1 = -r0 / b
            c2 = -j0 + (-r0 * (a-lambda1))/b
            def r_equation_dependent_eigenvectors(t):
                e1 =  cmath.e ** (lambda1 * t)
                return (c1 * (-b)) * e1 + c2 * ((-b) * t * e1)
            def j_equation_dependent_eigenvectors(t):
                e1 =  cmath.e ** (lambda1 * t)
                return (c1 * (a - lambda1)) * e1 + c2 * (((a-lambda1) * t * e1) - e1)
            return r_equation_dependent_eigenvectors, j_equation_dependent_eigenvectors
    

    return None, None


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
    solve_RJ(2.1601, 3.7439, 5.6771, -2.5837, -2, 3)

if __name__ == '__main__':
    main()

