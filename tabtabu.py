import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class TS:
    
    def __init__(self,MAX_GEN,length,N,Num,b_0=138,alpha=0.5,beta=0.7,pi=[],ri=[],C_max=0.0,R_max=0.0):
        '''
        MAX_GEN: Maximum number of iterations
        N: The number of candidate sets selected from the field
        length: the length of the taboo table
        Num: the number of processes, that is, the length of the code
        b_0: ambiguity threshold
        alpha, beta: penalty coefficient
        pi: the primary key value of the processing time in the original sequence
        ri: the ambiguity of the processing time of the original sequence
        C_max: The maximum value of the total processing time in all rows
        R_max: The maximum value of the total processing time ambiguity in all the arrangement methods
        '''
        self.MAX_GEN = MAX_GEN
        self.length = length
        self.N = N
        self.Num = Num
        self.b_0 = b_0
        self.alpha = alpha
        self.beta = beta
        self.pi = pi
        self.ri = ri
        self.C_max = C_max
        self.R_max = R_max
        self.P_pi = []
        self.neighbor = [] #Neighbor
        
        self.Ghh = [] #Current best encoding
        self.current_fitness = 0.0 #The fitness value of the current best encoding
        self.fitness_Ghh_current_list = [] #The current best coded fitness value list
        self.Ghh_list = [] #The current best coded list
        
        self.bestGh = [] #The best encoding
        self.best_fitness = 0.0 #Best coded fitness value
        self.best_fitness_list = [] #Best coded fitness value list
        self.tabu_list = np.random.randint(0,1,size=(self.length,self.Num)).tolist() #Initialize taboo table
                
    #Generate initial solution
    def InitialSolution(self):
        self.Ghh = (np.argsort(self.ri)+1).tolist() #The initial solution is to arrange the workpieces in non-decreasing order of their processing time ambiguity
       
    #2-opt Neighborhood exchange, get neighbors
    def swap(self):
        for i in range(len(self.Ghh)-1):
            for j in range(i+1,len(self.Ghh)):
                temp = self.Ghh.copy()
                temp[i],temp[j] = temp[j],temp[i]
                self.neighbor.append(temp)
        print(self.neighbor)
    #Judging whether a code is in the taboo table
    def judgment(self,GN=[]):
        #GN: To determine whether the exchange operation is in the taboo table.
        flag = 0 #Indicates that this code is not in the taboo table
        for temp in self.tabu_list:
            temp_reverse = []
            for i in reversed(temp):
                temp_reverse.append(i)
            if GN == temp or GN == temp_reverse:
                flag = 1 #Indicates that this code is in the taboo table
                break
        return flag
    
    #Change taboo list
    def ChangeTabuList(self,GN=[],flag_ = 1):
        #GN: To insert the new exchange operation of the taboo table, GN=[1,2]
        #flag_: used to determine whether the principle of contempt is satisfied, flag_ = 1 means the principle of contempt is satisfied
        if flag_ == 0:
            self.tabu_list.pop() #pop the last code
            self.tabu_list.insert(0,GN) #Insert a new code at the start position
        if flag_ == 1:
            for i, temp in enumerate(self.tabu_list):
                temp_reverse = []
                for j in reversed(temp):
                    temp_reverse.append(j)
                if GN == temp or GN == temp_reverse:
                    self.tabu_list.pop(i)
                    self.tabu_list.insert(0,GN)
        
    #Fitness function (evaluation function)
    def fitness(self,GN=[]):
        #GN: To calculate the encoding of the fitness function
        fitness_pi_ij = 0.0
        #Calculate the primary key value corresponding to the GN code of the process
        p_pi_ij = 0.0
        for i in range(self.Num):
            p_pi_ij = p_pi_ij + (self.Num-i)*self.pi[GN[i]-1]
        #Calculate the ambiguity corresponding to the GN code of the adding process
        r_pi_ij = 0.0
        for i in range(self.Num):
            r_pi_ij = r_pi_ij + (self.Num-i)*self.ri[GN[i]-1]
        #Calculate fitness
        if r_pi_ij <= self.b_0:
            fitness_pi_ij = 2*self.C_max + 1-p_pi_ij
        elif self.b_0 <r_pi_ij and r_pi_ij <=((1-self.alpha)*self.R_max+self.alpha*self.b_0):
            fitness_pi_ij = 2*self.C_max + 1-p_pi_ij -self.beta*self.C_max*(r_pi_ij-self.b_0)/((1-self.alpha)*(self.R_max-self.b_0))
        elif r_pi_ij >=((1-self.alpha)*self.R_max+self.alpha*self.b_0):
            fitness_pi_ij = self.C_max + 1-p_pi_ij + (1-self.beta)*self.C_max*(self.R_max-r_pi_ij)/(self.alpha*(self.R_max-self.b_0))
        return fitness_pi_ij
    
    def solver(self):
        #initialization
        self.InitialSolution() #Generate the current best encoding self.Ghh
        self.current_fitness = self.fitness(GN = self.Ghh) #self.Ghh's fitness value
        
        self.bestGh = self.Ghh #Copy self.Ghh to the best code self.bestGh
        self.best_fitness = self.current_fitness #Best fitness value
        self.best_fitness_list.append(self.best_fitness)
          
        self.Ghh_list.append(self.Ghh.copy()) ##Update the current list of best codecs
        self.fitness_Ghh_current_list.append(self.current_fitness) #Update the current best fitness value list
        
        step = 0 #Current iteration steps
        while(step<=self.MAX_GEN):
            self.swap() #Generate a two-dimensional list of neighbors self.neighbor, remember that you need to blank it later
            #Calculate the fitness function value of each neighbor
            fitness = []
            for temp in self.neighbor:
                fitness_pi_ij = self.fitness(GN = temp) #The fitness of a processing sequence passed in
                fitness.append(fitness_pi_ij)
            #Arrange the order of candidates according to the fitness function value from large to small
            temp = np.argsort(fitness).tolist()
            fitness_sort = [] #Fitness sorted value
            for i in temp:
                fitness_sort.append(fitness[len(fitness)-1-i])
            neighbor_sort = [] #Sorting neighbors in candidate order according to the fitness function value from large to small, the first neighbor has the largest fitness function value
            for i in range(len(temp)):
                neighbor_sort.append(self.neighbor[temp[len(temp)-1-i]])
            self.neighbor = [] #Empty the neighbor's two-digit list for next use
            
            
            neighbor_sort_N = neighbor_sort[:self.N] #Select the first N codes with the best adaptation value in the neighbor
            fitness_sort_N = fitness_sort[:self.N] #Select the first N fitness function values ​​with the best fitness value among neighbors
            
            
            m = 0
            for temp in neighbor_sort_N:
                GN = [] #The element used to install the exchange GN=[1,2] is the same as GN=[2,1]
                for i,temp_Ghh in enumerate(self.Ghh): #self.Ghh: current best encoding
                    if temp_Ghh != temp[i]:
                        GN.append(temp_Ghh)
                                
                flag = self.judgment(GN=GN) #Judging whether the exchange is in the taboo table
                if flag == 1: # indicates that this exchange is in the taboo table
                    #Judging whether the contempt criterion is met
                    if fitness_sort_N[m]>self.best_fitness: #Satisfy the rules of contempt
                        self.current_fitness = fitness_sort_N[m] #Update the current best fitness function value
                        self.fitness_Ghh_current_list.append(self.current_fitness) #Update the current best fitness function value list
                        self.Ghh = neighbor_sort_N[m] #Update the current best code
                        self.Ghh_list.append(self.Ghh.copy()) #Update the current list of best codecs
                        
                        
                        self.best_fitness = fitness_sort_N[m] #Update the best fitness function value
                        self.best_fitness_list.append(self.best_fitness)
                        self.bestGh = temp.copy() #Update the best code
                        #Update Taboo List
                        self.ChangeTabuList(GN=GN, flag_=1)
                        break
                    else:
                        m = m + 1
                else: #Indicates that this exchange is not in the taboo list
                    if fitness_sort_N[0] <self.current_fitness:
                        self.current_fitness = fitness_sort_N[0] #Update the current best fitness value
                        self.Ghh = neighbor_sort_N[0] #Update the current best code
                        self.Ghh_list.append(self.Ghh.copy()) #Update the current list of best codecs
                        self.fitness_Ghh_current_list.append(self.current_fitness) #Update the current best fitness function value list
                        #Update Taboo List
                        self.ChangeTabuList(GN=GN, flag_=0)
                        break
                    else:
                        self.current_fitness = fitness_sort_N[0] #Update the current best fitness value
                        self.Ghh = neighbor_sort_N[0] #Update the current best code
                        self.Ghh_list.append(self.Ghh.copy()) #Update the current list of best codecs
                        self.fitness_Ghh_current_list.append(self.current_fitness) #Update the current best fitness function value list
                        #Update Taboo List
                        self.ChangeTabuList(GN=GN, flag_=0)
                        if fitness_sort_N[0]>self.best_fitness:
                            self.best_fitness = fitness_sort_N[0] #Update the best fitness function value
                            self.best_fitness_list.append(self.best_fitness)
                            self.bestGh = neighbor_sort_N[0].copy() #Update the best code
                        break
            P_pi = 0
            for i in range(self.N):
                P_pi = P_pi + (self.N-i)*self.pi[self.Ghh[i]-1]
            self.P_pi.append(P_pi)
                             
            step = step + 1
            
            
if __name__ =='__main__':
    df = pd.read_csv('example.txt',sep='',index_col=['i'])
    pi = df.values[:,0].tolist()
    ri = df.values[:,1].tolist()
    pi_drop = sorted(pi,reverse=True)
    ri_drop = sorted(ri,reverse=True)
    C_max = 0.0 #The maximum value of the total processing time in all arrangements
    Len = len(pi_drop)
    for i in range(Len):
        C_max = C_max + (Len-i)*pi_drop[i]
    R_max = 0.0 #The maximum value of the ambiguity of total processing time in all arrangement
    for i in range(Len):
        R_max = R_max + (Len-i)*ri_drop[i]
    
    ts = TS(MAX_GEN=60,length=5,N=Len,Num=Len,b_0=138,alpha=0.5,beta=0.7,pi=pi,ri=ri,C_max=C_max, R_max=R_max)
    ts.solver()
    
    print('Best fitness function value: %.2f'%ts.best_fitness)
    print('Best processing order:',ts.bestGh)
    P_pi = 0.0 #The total primary key value of the best processing sequence processing time
    R_pi = 0.0 #The total ambiguity of the best processing sequence processing time
    for i in range(Len):
        P_pi = P_pi + (Len-i)*pi[ts.bestGh[i]-1]
        R_pi = R_pi + (Len-i)*ri[ts.bestGh[i]-1]
    print('The total primary key value of the best processing sequence: %.2f'%P_pi)
    print('The total ambiguity value of the best processing sequence: %.2f'%R_pi)
    
    P= []
    
    for temp in ts.Ghh_list:
        P_PI = 0.0
        for i in temp:
            P_PI = P_PI + (Len-i)*pi[i-1]
        P.append(P_PI)
    
    #================================================ ==========================
    #
    # plt.plot(ts.fitness_Ghh_current_list,color='blue')
    # #plt.plot(P,color='yellow')
    # #plt.ylim(min(P),max(P))
    # plt.xlabel(r'Number of iteration steps$i$')
    # plt.ylabel(r'$fitness({\pi _i})$')
    # plt.title('The value of fitness function changes with the number of iteration steps')
    # plt.show()
    #================================================ ==========================
     
    plt.plot(ts.P_pi,color='blue')
    plt.xlabel(r'Number of iteration steps$i$')
    plt.ylabel(r'${\rm{P}}\left( {{\pi _i}} \right)$')
    plt.show()
