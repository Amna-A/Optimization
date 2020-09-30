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

def gradient_1(choice,x):
    h = 0.001
    if choice ==0:
        g_x = (toy_problem_0(x[0]+h, x[1])-toy_problem_0(x[0]-h, x[1]))/(2*h)
        g_y = (toy_problem_0(x[0], x[1]+h)-toy_problem_0(x[0], x[1]-h))/(2*h)
    elif choice ==1:
        g_x = (toy_problem_1(x[0]+h, x[1])-toy_problem_1(x[0]-h, x[1]))/(2*h)
        g_y = (toy_problem_1(x[0], x[1]+h)-toy_problem_1(x[0], x[1]-h))/(2*h)
    elif choice ==2:
        g_x = (toy_problem_2(x[0]+h, x[1])-toy_problem_2(x[0]-h, x[1]))/(2*h)
        g_y = (toy_problem_2(x[0], x[1]+h)-toy_problem_2(x[0], x[1]-h))/(2*h)
    elif choice ==3:
        g_x = (toy_problem_3(x[0]+h, x[1])-toy_problem_3(x[0]-h, x[1]))/(2*h)
        g_y = (toy_problem_3(x[0], x[1]+h)-toy_problem_3(x[0], x[1]-h))/(2*h)
    return np.array([g_x, g_y])

def hessian(choice,x):
    return nd.Hessian(lambda x: problem(choice,(x[0],x[1])))([x[0], x[0]])

import numpy.linalg 

def newton(choice, f, Df,Dff, start_point):

    z = []
    tolerence = 0.00001
    iterations = 100
    counter = 0
    step_size = 1
    
    #la.solve(Dff(choice,old_point), Df(choice,old_point))
    
    while step_size > tolerence and counter < iterations:
        
        if choice ==0:
            new_point = start_point - np.dot(np.linalg.pinv(0.04*Dff(choice,start_point)) , Df(choice,start_point))
        elif choice ==1 or choice ==2:
            new_point = start_point - np.dot(np.linalg.pinv(0.0001*Dff(choice,start_point)) , (0.0001*Df(choice,start_point)))
        else:
            new_point = start_point - np.dot(np.linalg.pinv(Dff(choice,start_point)) , Df(choice,start_point))
        start_point = new_point
        new_f =  f(choice,new_point)
        step_size = norm(new_f)
        counter+=1
        z.append(f(choice,new_point))
        
        if np.linalg.eig(Dff(choice,start_point)) == False:
            break
      
    print(f"value is: {new_f}, at: {new_point}")
