from numpy.linalg import norm
import numpy as np
import numdifftools as nd
import pandas as pd
import matplotlib.pyplot as plt

def problem(choice,x):
    
    if choice ==0:
        obj_func = toy_problem_0(x[0],x[1])
    elif choice==1:
        obj_func = toy_problem_1(x[0],x[1])
    elif choice==2:
        obj_func = toy_problem_2(x[0],x[1])
    elif choice==3:
        obj_func = toy_problem_3(x[0],x[1])
    return obj_func

def gradient(choice,x):
    if choice ==0:
        gx = nd.Gradient(toy_problem_0)(x,x)
    elif choice ==1:
        gx = nd.Gradient(toy_problem_1)(x,x)
    elif choice ==2:
        gx = nd.Gradient(toy_problem_2)(x,x)
    elif choice ==3:
        gx = nd.Gradient(toy_problem_3)(x,x)
    return gx[0][0]

def gradient_descent(choice, f, gxy, init_point, rate,color):
    
    x, z = [] , []
    precision = 0.0000001
    step_size = 1
    iterations = 10000
    counter = 0
    
    while step_size > precision and counter < iterations:
        previous_point = init_point.copy()
        init_point -= np.dot(rate , gxy(choice,init_point) )
        step_size = norm(init_point - previous_point)
        z.append(f(choice,init_point))
        x.append(init_point)
        counter += 1
    print()
    print(f"- The minimum point of {f.__name__}_{choice} is at: {init_point} at iteration no.{counter}")
    print(f"- The value is :{f(choice,init_point)}")
    
     #plotting:
    Df = pd.concat([pd.DataFrame(x),pd.DataFrame(z)] , axis = 1)
    Df.columns=["x","y","z"]    
    plt.plot(Df["z"], color=f'{color}')
    plt.title(f'{f.__name__}_{choice}')
    plt.xlabel('Iterations')
    plt.ylabel('Optimization Value')
    plt.grid(linewidth = "0.5")
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)