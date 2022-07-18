"""
# Author  : Jason Connie
# Updated : July 2022

Code to simulate the non-local CHSH game. 

A maximum classical winning probability is 0.75, 
yet quantum mechanics allows for around 0.85

See https://qiskit.org/textbook/ch-demos/chsh.html for a run down of the game, 
and a different simulation using the qiskit library.

"""

import numpy as np

# Our rotation operator
def R(theta):
    r1 = np.cos(theta)
    r2 = np.sin(theta)
    return np.array([[r1,-r2],[r2,r1]])


# Setting our primary bits to fairly generated random values, but they could be manually set as desired
a = np.random.randint(2)
b = np.random.randint(2)


# The measurement operators associated with the Z basis
Z_p = np.array([[1,0],[0,0]])
Z_n = np.array([[0,0],[0,1]])


# Choosing the appropriate angle of rotation for Alice and Bob's measurements
if (a==0):
    a_theta = 0
else:
    a_theta = -np.pi/4

if (b==0):
    b_theta = -np.pi/8
else:
    b_theta = np.pi/8


# Defining the measurement operators we will use, by rotating the Z basis
# Note: @ indicates matrix multiplication
MA_x0 = R(a_theta)@Z_p@R(-a_theta) # The measurement operator corresponding to x=0
MA_x1 = R(a_theta)@Z_n@R(-a_theta) # and to x=1

MB_y0 = R(b_theta)@Z_p@R(-b_theta) # The measurement operator corresponding to y=0
MB_y1 = R(b_theta)@Z_n@R(-b_theta) # and to y=1


# Our Bell state, in bra and ket form
state_ket = (1/np.sqrt(2))*np.array([[1],[0],[0],[1]])
state_bra = np.conj(state_ket).T


Success_prob = 0

# If (a=1, b=1), we want to see how often x and y disagree
if (a==1 and b==1):
    Success_prob += state_bra@np.kron(MA_x0, MB_y1)@state_ket
    Success_prob += state_bra@np.kron(MA_x1, MB_y0)@state_ket

# But otherwise, we want to see how often x and y agree
else:
    Success_prob += state_bra@np.kron(MA_x0, MB_y0)@state_ket
    Success_prob += state_bra@np.kron(MA_x1, MB_y1)@state_ket

print("The probability of winning the game is", round(Success_prob[0][0],7))