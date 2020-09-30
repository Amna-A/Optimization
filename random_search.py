import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def RandomSearch(func,RANGE,color):

    #initiallizing..step0
    x_0, y_0 = 13,17
    z_0 = func(x_0,y_0)
    
    #for plotting
    x_list , y_list , z = [] , [] , []
    
    #Step1,2,3..
    for _ in range(RANGE):
        k1 , k2 = np.random.uniform(-1,1, size = 2)
        x_new = x_0 + k1
        y_new = y_0 + k2
        z_new = func(x_new , y_new)
        x_list.append(x_new)
        y_list.append(y_new)
        z.append(z_new)
        if z_new < z_0:
            z_0 = z_new
            x_0,y_0 = x_new,y_new
        
            
    print(f"The minimum of '{func.__name__}' is {z_0} at ({x_0},{y_0})")
    
    #plotting
    Df = pd.concat([pd.DataFrame(x_list),pd.DataFrame(y_list),pd.DataFrame(z)] , axis = 1)
    Df.columns=["x","y","z"]    
    plt.plot(Df["z"],color=f'{color}')
    plt.xlabel('Range of Values')
    plt.ylabel('optimization Result')
    plt.title(f'{func.__name__}')
    plt.grid(linewidth = "0.4")
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
