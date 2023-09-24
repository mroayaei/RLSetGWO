# Deep Q-Network
# each solution (agent) is represented as a vector. Also, use set encoding for opertions
# actions select which neighborhood 
# reward: fitness change
# state: current solution + 3leaders
# operation: explore, exploit, random, crossover
# layer size: 24

 
#Imports
import random
import math
import time
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow import keras
from scipy.special import softmax
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam

def Neighborhood_DQN(pop_size,MaxIt,instance,NUM):

    #create Environment 
    #create defualt config
    # Model Configuration
    config = mc.Configuration()   
    graph = instance.graph
    # Setting the edge parameters    
    removed_edges = []
    for e in graph.edges():
        # we set the weight of every incoming edge of a node v to be equal to 1/dv,
        # where dv is the in-degree of node v.
        if(random.random() <= instance.edge_threshold):           
            config.add_edge_configuration("threshold", e, 1)
            #e['act_prob'] = instance.edge_threshold
        else:                  
            removed_edges.append(e)
    graph.remove_edges_from(removed_edges)
    
    print("program " +str(NUM) +" starting....")          
    dim = instance.budget    


    # maximization
    num_nodes = len(instance.graph.nodes())
    
    leader_pos = np.zeros((3,num_nodes), dtype=int)
    leader_score = np.zeros(3, dtype=int)
    leader_set = [None] *  3

    
    for i in range(3):
        leader_pos[i] =np.zeros(num_nodes, dtype=int)
        leader_score[i]= -1   

    
                
    #all items    
    
    nodes_list = sorted(list(instance.graph.nodes()))
    #Initialize the population
    Population = np.zeros((pop_size,num_nodes))       



    timerStart = time.time() 
    # finish criteria
    counter = 0 
    old_top_score = 0
  
    fitness_time = 0
    total_time = 0
    COVERGE_LIMIT = 30

    fitness = [None] * pop_size


    #Global Variables
    EPISODES = 20
    TRAIN_END = 0
    #Hyper Parameters
    def discount_rate(): #Gamma
        return 0.95

    def learning_rate(): #Alpha
        return 0.01

    def batch_size(): #Size of the batch used in the experience replay
        return 100


    
    #Create the agent
    # state size: current solution + three leaders
    nS = num_nodes + 3 * num_nodes
    # number of actions
    nA = 9  #
    dqn = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.01, 0.095)    
    batchsize = batch_size()
    PopulationSet = [None] * pop_size

    # test
    

    indices_init = set(range(num_nodes))
    for  i in range(pop_size):          
        #initialization
        #print("generating solution "+ str(i))            
        # select nodes w.r.t. probabilities 
        selected_seeds = random.sample(indices_init,k = dim)    
        PopulationSet[i]  = set([nodes_list[index] for index in selected_seeds])
        for node_index in selected_seeds:
            Population[i][node_index] = 1 
        indices_init = indices_init - set(np.nonzero(Population[i])[0])
        if (len(indices_init) < 3 * dim):
            # refill set_seq
            indices_init = set(range(num_nodes))

                                           
        # Calculate objective function for each wolf
        seed = [nodes_list[index] for index in list(np.nonzero(Population[i])[0])]        
        config.add_model_initial_configuration("Infected",seed)            
        # Model selection        
        model = ep.IndependentCascadesModel(graph)                                
        model.set_initial_status(config)        
        # Simulation execution
        iterations = model.iteration_bunch(30)     
        status = model.status
        fitness[i] = sum([1 for x in list(status.values()) if x > 0])
        
       
                
        # Update Alpha, Beta, Delta
        if fitness[i] > leader_score[0] :
            # Update alpha
            leader_score[0] = fitness[i] 
            leader_pos[0] = Population[i].copy()
            leader_set[0] = PopulationSet[i].copy()
        
        
        if (fitness[i] < leader_score[0] and fitness[i] > leader_score[1]):
            # Update beta
            leader_score[1] = fitness[i]  
            leader_pos[1] = Population[i].copy()
            leader_set[1] = PopulationSet[i].copy()
        
        
        if (fitness[i] < leader_score[0] and fitness[i] < leader_score[1] and fitness[i] > leader_score[2]): 
            # Update delta
            leader_score[2] = fitness[i]
            leader_pos[2] = Population[i].copy()
            leader_set[2] = PopulationSet[i].copy()
        
        # ensure beta and delta will not be -1
        if leader_score[1] == -1:
            leader_score[1] = leader_score[0]
            leader_pos[1]   = leader_pos[0].copy()
            leader_set[1]   = leader_set[0].copy()    
        
        if leader_score[2] == -1:
            leader_score[2] = leader_score[0]
            leader_pos[2]   = leader_pos[0].copy()
            leader_set[2]   = leader_set[0].copy()  
    
    if (leader_score[0]) < 60:
        #print("aplha is bigger than 60")
        return 60 , 0, 0
        
    #Training
    it=0
    for it in range(MaxIt):  
        #print("iteration " + str(it)+"  starts....")                
        #todo: tot_rewards in or out of outer loop        
        fitness_new = [None] * pop_size
        state =  [None] * pop_size
        action = [None] * pop_size        
        reward = [None] * pop_size
        nstate = [None] * pop_size

        crossover_counter1 = 0
        crossover_counter2 = 0
        crossover_counter3 = 0
        random_counter     = 0
        mutation_counter   = 0
        inverse_counter    = 0
        displace_counter   = 0
        swap_counter       = 0
        insert_counter     = 0
    

        for i in range(pop_size):           
            current_state = list(Population[i]) + list(leader_pos[0]) + list(leader_pos[1]) + list(leader_pos[2])
            state[i] = np.reshape(current_state, [1, nS]) # Resize to store in memory to pass to .predict                                              
            action[i]  = dqn.action(state[i],nA)      
            done = False




            # check if action is valid
            if action[i] in {0}:
                # set-explore
                PopulationSet[i],indices_init = set_explore(0.2, PopulationSet[i], leader_set, indices_init, num_nodes,nodes_list)
                Population[i]    = set_to_list(PopulationSet[i], num_nodes,nodes_list)

            if action[i] in {1}:
                # set-exploit
                PopulationSet[i] = set_exploit(0.2, PopulationSet[i], leader_set)
                Population[i]    = set_to_list(PopulationSet[i], num_nodes, nodes_list)
            
            if action[i] in {2} :
                random_counter += 1   
                #random
                Population[i],indices_init = random_wolf(dim,num_nodes,indices_init,nodes_list)
                PopulationSet[i] = list_to_set(Population[i],nodes_list)

            if action[i] in {3} :
                crossover_counter1 += 1   
                #crossover 
                partner_index = random.randint(0,2)
                #crossover of pop[i] with leader
                Population[i] = crossover(Population[i],leader_pos[partner_index],dim)     
                PopulationSet[i] = list_to_set(Population[i],nodes_list)
                                                     

            elif action[i] in {4}:       
                #mutation 
                mutation_counter += 1
                Population[i] = mutation(Population[i])
                PopulationSet[i] = list_to_set(Population[i],nodes_list)

            elif action[i] in {5}:
                # pairswap: exploration
                swap_counter += 1
                Population[i] = pair_swap(Population[i])
                PopulationSet[i] = list_to_set(Population[i],nodes_list)

            elif action[i] in {6}:
                # insertion: exploration
                insert_counter += 1
                Population[i] = insersion(Population[i])
                PopulationSet[i] = list_to_set(Population[i],nodes_list)
            
            elif action[i] in {7}:
                # reverse: exploration
                inverse_counter += 1
                Population[i] = inversion(Population[i])
                PopulationSet[i] = list_to_set(Population[i],nodes_list)

            elif action[i] in {8}:
                # dispace: exploration
                displace_counter += 1
                Population[i] = displace(Population[i]) 
                PopulationSet[i] = list_to_set(Population[i],nodes_list)
            
 
            
            seed = [nodes_list[index] for index in list(np.nonzero(Population[i])[0])]        
            config.add_model_initial_configuration("Infected",seed)            
            # Model selection        
            model = ep.IndependentCascadesModel(graph)                                
            model.set_initial_status(config)        
            # Simulation execution
            iterations = model.iteration_bunch(30)     
            status = model.status
            fitness_new[i] = sum([1 for x in list(status.values()) if x > 0])  

            if fitness_new[i] - fitness[i] > 0:            
                reward[i] = +2
            else:
                reward[i] = -1

            fitness[i] = fitness_new[i] 
            # Update Alpha, Beta, Delta, gamma
            if fitness[i] > leader_score[0] :
                # Update alpha
                leader_score[0] = fitness[i] 
                leader_pos[0] = Population[i].copy()
                leader_set[0] = PopulationSet[i].copy()
            
            
            if (fitness[i] < leader_score[0] and fitness[i] > leader_score[1]):
                # Update beta
                leader_score[1] = fitness[i]  
                leader_pos[1] = Population[i].copy()
                leader_set[1] = PopulationSet[i].copy()
            
            
            if (fitness[i] < leader_score[0] and fitness[i] < leader_score[1] and fitness[i] > leader_score[2]): 
                # Update delta
                leader_score[2] = fitness[i]
                leader_pos[2]= Population[i].copy()
                leader_set[2] = PopulationSet[i].copy()
        
        # ensure beta and delta will not be -1
        if leader_score[1] == -1:
            leader_score[1] = leader_score[0]
            leader_pos[1]   = leader_pos[0].copy()
            leader_set[1]   = leader_set[0].copy()    
        
        if leader_score[2] == -1:
            leader_score[2] = leader_score[0]
            leader_pos[2]   = leader_pos[0].copy()
            leader_set[2]   = leader_set[0].copy()  

        
        for j in range(pop_size):
            nstate[j] = list(Population[j]) + list(leader_pos[0]) + list(leader_pos[1]) + list(leader_pos[2])
            nstate[j] = np.reshape(nstate[j], [1, nS])
            dqn.store(state[j], action[j], reward[j], nstate[j], False) # Resize to store in memory to pass to .predict
             
        
        #update model after each iteration
        dqn.experience_replay(batchsize)
        #state = nstate  

        #Convergence_curve[it] = Alpha_score        
        print(['At iteration '+ str(it)+ ' the best fitness is '+ str(leader_score[0])+ '  time: ' +str(time.time() - timerStart)])
        # print('crossover1: '+str(crossover_counter1))
        # print('crossover2: '+str(crossover_counter2))
        # print('random: '+str(random_counter))
        # print('mutation: '+str(mutation_counter))
        # print('inverse: '+str(inverse_counter))
        # print('displace: '+str(displace_counter))
        # print('swap: '+str(swap_counter))
        # print('insert: '+str(insert_counter))
        if leader_score[0] == old_top_score:
            counter +=1
        else:
            counter = 0
        if counter == COVERGE_LIMIT:
            break
        old_top_score = leader_score[0]
    
    result = leader_score[0]
    total_time = time.time() - timerStart       
    #print best solution    
  
    return result, total_time, it+1 
      

# Deep Q-Network Class
class DeepQNetwork():
    def __init__(self, states, actions, alpha, gamma, epsilon,epsilon_min, epsilon_decay):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma
        #Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()      
        self.loss = []

    
    def decay_epsilon(self): 
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # This builds and returns the NN
    def build_model(self):
        model = keras.Sequential() #linear stack of layers https://keras.io/models/sequential/
        model.add(keras.layers.Dense(24, input_dim=self.nS, activation='relu')) #[Input] -> Layer 1

        model.add(keras.layers.Dense(24, activation='relu')) #Layer 1 -> 2
        model.add(keras.layers.Dense(24, activation='relu')) #Layer 2 -> 3
        model.add(keras.layers.Dense(self.nA, activation='linear')) #Layer 3 -> [output]
        #   Size has to match the output (different actions)
        #   Linear activation on the last layer
        model.compile(loss='mean_squared_error', #Loss function: Mean Squared Error
                     optimizer=keras.optimizers.Adam(lr=self.alpha)) #Optimaizer: Adam (Feel free to check other options)
        return model

    #  This generates the action.
    def action(self, state, nactions):     
        if np.random.rand() <= self.epsilon:       
            return random.choice(range(nactions))
        action_vals = self.model.predict(state) #Exploit: Use the NN to predict the correct reomve action from this state                            
        return np.argmax(action_vals[0])
         
        
   
    def action_boltzmann(self, state,temp):
        temp=10000
        action_vals = self.model.predict(state) #Exploit: Use the NN to predict the correct reomve action from this state                            
        Q_prob = softmax(action_vals[0]/temp)                    
        action_value = np.random.choice(Q_prob,p=Q_prob)
        action = np.argmax(Q_prob == action_value)
        return action
    
    #  This generates the action during testing. We want to 100% exploit
    def test_action(self, state): #Exploit
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    #  This places the observables in memory
    def store(self, state, action, reward, nstate, done):
        #Store the experience in memory
        self.memory.append( (state, action, reward, nstate, done) )

    

    def experience_replay(self, batch_size):
        #Execute the experience replay
        batch_size = len(self.memory)
        minibatch = random.sample( self.memory, batch_size) #Randomly sample from memory

        #Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = np.array(minibatch)
        st = np.zeros((0,self.nS)) #States
        nst = np.zeros( (0,self.nS) )#Next States
        for i in range(len(np_array)): #Creating the state and next state np arrays
            st = np.append( st, np_array[i,0], axis=0)
            nst = np.append( nst, np_array[i,3], axis=0)
        st_predict = self.model.predict(st) #Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            #Predict from state
            nst_action_predict_model = nst_predict[index]
            if done == True: #Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:   #Non terminal
                target = reward + self.gamma * np.amax(nst_action_predict_model)
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        #Reshape for Keras Fit
        x_reshape = np.array(x).reshape(batch_size,self.nS)
        y_reshape = np.array(y)
        epoch_count = 1 #Epochs is the number or iterations
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        #Graph Losses
        for i in range(epoch_count):
            self.loss.append( hist.history['loss'][i])
        #Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.memory.clear()


# function for set-exploit
# extent: percentage of exploit
def set_exploit(extent: float,wolf_set:set,leader_set:list):
    leaders_set = set().union(*leader_set)
    new_items = leaders_set - wolf_set
    old_items = wolf_set   - leaders_set
    D = len(old_items)
    selecteds_num = abs(math.ceil(extent * D))
    selecteds_num = min(selecteds_num, len(old_items), len(new_items))
    wolf_set.difference_update(random.sample(old_items,k=selecteds_num))
    wolf_set = wolf_set.union(random.sample(new_items, k=selecteds_num))
    return wolf_set


# function for set-explore
# input: extent: percentage of explore
def set_explore(extent:int,wolf_set:set,leader_set:list,indices_init:set,num_nodes:int,nodes_list):    
    leaders_set = set().union(*leader_set)
    new_items = set([nodes_list[index] for index in indices_init]) - (wolf_set.union(*leader_set))
    old_items = set().union(leaders_set).intersection(wolf_set)    
    D = len(wolf_set - leaders_set)
    selecteds_num = math.ceil(abs(extent * D))
    selecteds_num = min(selecteds_num, len(old_items), len(new_items))
    news = set()
    if(selecteds_num != 0):
        wolf_set.difference_update(random.sample(old_items,k=selecteds_num))
        news = set(random.sample(new_items, k=selecteds_num))
        wolf_set = wolf_set.union(news)  
    indices_init = indices_init - news   
    if (len(indices_init) < 3 * len(wolf_set)):
        # refill set_seq
        indices_init = set(range(num_nodes))
    return wolf_set, indices_init





# function for implementing the single-point crossover
def crossover(l:list, q:list, dim:int):
    length = len(l)        
    q=list(q)
    l=list(l)
    l_new = [None] * length
    # generating the random number to perform crossover
    k = random.randint(1, length-2)           
    # random crossover
    if(random.random() < 0.5):
        l_new = l[0:k] + q[k:length]  
    else:
        l_new = q[0:k] + l[k:length]        
    
    #check budget
    while len(np.nonzero(l_new)[0]) > dim:
        l_new[random.sample(set(np.nonzero(l_new)[0]),k=1)[0]] = 0
    
    while len(np.nonzero(l_new)[0]) < dim:
        indices = [i for i, x in enumerate(l_new) if x == 0]
        l_new[random.sample(indices,k=1)[0]] = 1    
    return l_new

# function for implementing the random function
def random_wolf(dim:int, num_nodes:int, indices_init:set, nodes_list:list):
    wolf = np.zeros(num_nodes) 
    selected_seeds_indices = random.sample(indices_init,k = dim)    
    for node_index in selected_seeds_indices:
        wolf[node_index] = 1
    indices_init = indices_init - set(np.nonzero(wolf)[0])
    if (len(indices_init) < dim):
        # refill set_seq
        indices_init = set(range(num_nodes)) 
    
    return wolf,indices_init


# function for implementing the single-point crossover
def mutation(l: list):
    l = list(l)
    length = len(l)
    l_new = [None] * length        
    mut_size = 1
    # generating the random number to perform crossover
    nonzero_indices = list(random.sample(set(np.nonzero(l)[0]),k=mut_size))
    zero_indices =  list(random.sample (set(range(0,length)) - set(nonzero_indices),k=mut_size))
    l_new = l
    for i in range(mut_size):
        l_new[nonzero_indices[i]] = 0 
        l_new[zero_indices[i]] = 1

    return l_new


# function for implementing pair swap
def pair_swap(l: list):
    l = list(l)
    length = len(l)
    l_new = [None] * length   
    l_new = list(l)
    # select which indices to change
    indices = random.sample(range(length), k=2)
    temp = l_new[indices[0]]
    l_new[indices[0]] = l_new[indices[1]]
    l_new[indices[1]] = temp
    
    return l_new

# function for implementing inverse
def inversion(l: list):
    l = list(l)
    length = len(l)
    l_new = [None] * length   
    l_new = list(l)
    # select which indices to change
    indices = random.sample(range(length), k=2)
    start = min(indices)
    end   = max(indices)  
    l_new[start:end+1] = l_new[start:end+1][::-1]    
    
    return l_new

# function for implementing insertion
def insersion(l: list):
    l = list(l)
    length = len(l) 
    l_new = list()  
    # select which indices to change
    indices = random.sample(range(length), k=2)
    start = min(indices)
    end   = max(indices)       
    insert_value = l[end]
    l.pop(end)    
    l_new = l[0:start] + [insert_value] + l[start:]
    
    return l_new



# function for implementing displacement
def displace(l: list):
    l = list(l)
    length = len(l)
    l_new = list()
    # select which indices to change
    indices = random.sample(range(length), k=3)
    start = min(indices)
    end1  = max(indices)
    indices.remove(start)
    indices.remove(end1)
    end2  = indices[0]

    l_new = l[0:start] + [l[end2]] + [l[end1]]
    l.pop(end1)
    l.pop(end2)
    l_new = l_new + l[start:] 

    return l_new

def list_to_set(wolf_list:list, nodes_list):
    wolf_set = [nodes_list[index] for index in list(np.nonzero(wolf_list)[0])]
    return set(wolf_set)

def set_to_list(wolf_set:set, num_nodes: int, nodes_list:list):
    wolf_list = np.zeros(num_nodes)    
    for element in (wolf_set):
        wolf_list[nodes_list.index(element)] = 1
    return wolf_list


