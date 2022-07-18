"""
# Author  : Jason Connie
# Updated : July 2022

Code to simulate the OCB game as presented in the paper "Quantum correlations with no causal order" by Oreshkov, Costa and Brukner.

This game gives rise to (hypothetical) non-causal correlations, defying classical intuitions about time.
This behaviour is comparable to the non-local correlations of the CHSH game, with how entanglement can defy classical intuitions about space.

"""

# a is Alice's random bit (fairly generated)
# b is Bob's random bit   (fairly generated)
# x is Alice's prediction for what b is
# y is Bob's prediction for what a is

# bprime dictates whether Alice or Bob needs to predict the other's bit, and must be fairly generated
# - bprime=1 means y=a is the goal, corresponding to Bob needing to predict Alice's bit
# - bprime=0 means x=b is the goal, corresponding to Alice needing to predict Bob's bit



# We import the needed modules
import qcausal as q
import numpy as np


# We set up the Pauli matrices, for convenience
I = q.I
X = q.X
Y = q.Y
Z = q.Z


""" Alice's Strategy """
def M_A(x,a):
    term1 = (1/2)*( I + ((-1)**x)*Z )
    term2 = (1/2)*( I + ((-1)**a)*Z )
    return q.tensor(term1,term2)  



""" Bob's Strategy """
def M_B(y,b,bprime):    
    # If bprime=1, Bob wishes to receive Alice's bit 'a' before making his prediction
    if (bprime==1):
        term1 = (1/2)*( I + ((-1)**y)*Z )
        term2 = np.array([[1,0],[0,0]])        # Could be any random density matrix
        return q.tensor(term1, term2)
    
    # If bprime=0, Bob wishes for Alice to receive his bit 'b' before she makes her prediction
    elif (bprime==0):
        term1 = (1/2)*( I + ((-1)**y)*X )
        term2 = (1/2)*( I + ((-1)**(b+y))*Z )
        return q.tensor(term1,term2)
    


# The OCB process matrix we use to win the game!
W = (1/4)*(q.tensor(I,I,I,I) + (1/np.sqrt(2))*(q.tensor(I,Z,Z,I)+q.tensor(Z,I,X,Z)))


print("~ THE AMAZING QUANTUM GAME SIMULATOR! ~") 
print("Have fun messing with the fabric of time","\n")


# We check the outcome of the game over all four possible combinations of 'a' and 'b',
# as well as for both possible values of 'bprime' in each iteration, to make sure 
# Alice and Bob ALWAYS win more than 0.75 of the time regardless of the bits generated.
for a in range(2):
    for b in range(2):
                
        print("a=",a,", b=",b,sep="")
        
        
        # Playing the game, where Bob needs to guess 'a', so we must optimize for y=a
        bprime=1
        
        MA = 0
        for x in range(2):
            MA += M_A(x,a)
            
        MB = M_B(y=a, b=b, bprime=bprime)
        
        Bob_Success_Prob = q.trace(q.tensor(MA, MB)@W) # The success probability of y=a, if bprime=1
        print("Given b'=1, prob(y=a)=",np.round(Bob_Success_Prob,3),sep="")
        
        
        # Playing the game, where ALice needs to guess 'b', so we must optimize for x=b
        bprime=0
        
        MB= 0       
        for y in range(2):
            MB += M_B(y,b,bprime)
        
        MA = M_A(x=b, a=a)
        
        Alice_Success_Prob = q.trace(q.tensor(MA,MB)@W) # The success probability of x=b, if bprime=0
        print("Given b'=0, prob(x=b)=", np.round(Alice_Success_Prob,3), "\n", sep="")

