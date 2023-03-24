#Implementation imperative (Lee and Puranam, 2016)

#Importing modules
#from IPython import get_ipython
#get_ipython().magic('reset -sf')
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import datetime


STARTINGTIME = datetime.datetime.now().replace(microsecond=0)

#####################################################################################################
# SET SIMULATION PARAMETERS HERE
T = 500 #number of periods to simulate the model
NP = 1000 #number of pairs of agents

# Task environments
M = 10 #the number of possible actions
Alpha = 2 #Parameter for beta distribution
Beta = 2 #Parameter for beta distribution

#agent learning parameters
Tau_M = 0.01 #Top-down exploration by managers
Tau_S = 0.01 #Inverse of implementation precision
Lambda = 0 #Imperfect communication
Mu = 1 #Observation error

nchoice_manager = np.ones((M)) # To keep the number of trials for each action for averaging
nchoice_subordinate = np.ones((M)) # To keep the number of trials for each action for averaging
att_manager=np.zeros((M)) # Belief of manager
att_subordinate=np.zeros((M)) # Belief of subordinate
######################################################################################################

#Defining functions
def environment(a, b, dim): #Construct task environments
    r = np.random.beta(a, b, size = (dim))
    return r

def softmax(attraction,temp,dim): #softmax action selection
    prob=np.zeros((dim))
    denom=0
    i=0
    while i<dim:
        denom=denom + math.exp(attraction[i]/temp)
        i=i+1
    roulette=random.random()
    i=0
    p=0
    while i<dim:
        prob[i]=(math.exp(attraction[i]/temp))/denom
        p = p + prob[i]
        if p>roulette:
            choice = i
            return choice
            break #stops computing probability of action selection as soon as cumulative probability exceeds roulette
        i=i+1

def averaging(attraction, ntrials, action, pay): #initialize beliefs
    attraction[action] = (ntrials / (
            ntrials + 1)) * attraction[action] + ( 1 / (ntrials + 1)) * pay
    return attraction

def influence(attraction, order, dim): #initialize beliefs
    vector_order=np.zeros((dim))
    vector_order[order] = 1
    attraction = Lambda * attraction + (1-Lambda)*vector_order
    return attraction

def initialrepresentation(dim): #initialize beliefs
    r = np.zeros((dim))
    return r


#SIMULTAION IS RUN HERE
result_org=np.zeros((T,3))

for a in range(NP):
    # Initialize task environment and beliefs
    E = environment(Alpha, Beta, M)
    att_manager = initialrepresentation(M)
    att_subordinate = initialrepresentation(M)
    nchoice_manager = np.ones((M))
    nchoice_subordinate = np.ones((M))
    cumperf = 0
    for t in range(T):
        result_org[t, 0] = t

        # Manager chooses an action
        action_manager = softmax(att_manager, Tau_M, M)

        # Subordinate is influenced by manager's order
        att_subordinate = influence(att_subordinate, action_manager, M)

        # Subordinate chooses an action to implement
        action_subordinate = softmax(att_subordinate, Tau_S, M)

        # Observe payoff from an implemented action
        payoff = E[action_subordinate] + np.random.standard_normal()
        cumperf += payoff
        result_org[t, 1] += payoff / NP
        result_org[t, 2] += cumperf / NP

        # Subordinate learns from feedback
        att_subordinate = averaging(att_subordinate, nchoice_subordinate[action_subordinate], action_subordinate, payoff)
        nchoice_subordinate[action_subordinate] = nchoice_subordinate[action_subordinate] + 1

        # Manager learns from feedback
        pr = random.random()
        if pr >= Mu:
            att_manager = averaging(att_manager, nchoice_manager[action_subordinate], action_subordinate, payoff)
            nchoice_manager[action_subordinate] = nchoice_manager[action_subordinate] + 1
        else:
            incorrect_action = int(random.random() * M)
            while (incorrect_action == action_subordinate):
                incorrect_action = int(random.random() * M)
            att_manager = averaging(att_manager, nchoice_manager[incorrect_action], incorrect_action, payoff)
            nchoice_manager[incorrect_action] = nchoice_manager[incorrect_action] + 1

#WRITING RESULTS TO CSV FILE   
filename = ("Implementation Imperative"+"_tauM="+str(Tau_M) + "_tauS=" + str(Tau_S)+"_lambda="+str(Lambda)+"_mu="+str(Mu)+'.csv')
with open (filename,'w',newline='')as f:
    thewriter=csv.writer(f)
    thewriter.writerow(['Period', 'Performance', 'Cumulative Performance'])
    for values in result_org:
        thewriter.writerow(values)
    f.close()  

##PRINTING END RUN RESULTS
print ("Final Performance: "+str(result_org[T-1,2]))

ENDINGTIME = datetime.datetime.now().replace(microsecond=0)
TIMEDIFFERENCE = ENDINGTIME - STARTINGTIME
#print 'Computation time:', TIMEDIFFERENCE    
    
