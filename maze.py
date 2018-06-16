import numpy as np


def maze_7x5(a=-1.0, b=-5.0, goal=15.0):

    nx, ny = 7, 5

    S = np.zeros([nx, ny]) + a

    # blocks
    S[1,3] = b
    S[2,1:3] = b
    S[4,4:] = b
    S[4,:2] = b
    #
    S[1,4] = b
    S[6,3] = b

    #
    S[6,1] = goal

    return S




def maze_13x13(a=-1.0, b=-5.0, goal=15.0):

    nx, ny = 13, 13

    S = np.zeros([nx, ny]) - a
    
    S[1,4] = b
    S[2,2] = b
    S[4,4:8] = b
    S[4,:2] = b
    
    S[1:3,8] = b
    S[3,10:] = b
    
    S[7,8:12] = b
    
    S[6:12,2] = b
    
    # 
    S[:1,4] = b
    S[5:8,4] = b
    S[7:,6] = b

    #
    S[12,11] = goal

    return S



    
    
    
    


