# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:30:51 2022

@author: uie70742
"""

import numpy as np
from matplotlib import pyplot as plt
import random 
import math
import copy
import sys

class Particle:
    def __init__(self, fitness, dim, minx, maxx, seed):
        self.rnd = random.Random(seed)
        
        # inicializar la posición de la partícula xi(t)
        self.position = [0.0 for i in range(dim)]
        
        # inicializar la posición de la partícula vi(t)
        self.velocity = [0.0 for i in range(dim)]
        
        # inicializar las mejores posiciones de la partícula \hat{x}_i (t)
        self.best_part_pos = [0.0 for i in range(dim)]
        
        for i in range(dim):
            self.position[i] = ((maxx-minx)*self.rnd.random() + minx)
            self.velocity[i] = ((maxx-minx)*self.rnd.random() + minx)
        
        #Evaluar la función de costo
        self.fitness = fitness(self.position)
        
        self.best_part_pos = copy.copy(self.position) # inicialización de la mejor posición
        self.best_part_fitnessVal = self.fitness     #inicialización de la mejor evalucaion de la función de costo
        
    # algoritmo pso(particle swarm optimization)
    def pso(fitness, max_iter, n, dim, minx, maxx):
        ## Hypermarámetros
        #inercia
        w=0.72
        # factor cognitivo
        c1=1.5
        # factor social
        c2=1.5
        
        rnd = random.Random(0)
        
        swarm = [Particle(fitness, dim, minx, maxx, i) for i in range(n)]
        
        best_swarm_pos = [0.0 for i in range(dim)] # g(t)
        best_swarm_fitnessVal = sys.float_info.max # 
        
        for i in range(n):
            if swarm[i].fitness < best_swarm_fitnessVal:
                best_swarm_fitnessVal = swarm[i].fitness
                best_swarm_pos = copy.copy(swarm[i].position)
        
        # loop principal
        Iter = 0
        best_swarm_pos_hist ={}
        best_swarm_fitnessVal_hist ={}
        
        while Iter < max_iter:
            
            best_swarm_pos_hist[Iter] = best_swarm_pos
            
            best_swarm_fitnessVal_hist[Iter] = best_swarm_fitnessVal
            
            if Iter %10 ==0 and Iter>1:
                print(f"Iter = {Iter}, best position {np.round(best_swarm_pos,3)}, best_fitness_val= {np.round(best_swarm_fitnessVal,3)}")
            
            for i in range(n): #Iterando por cada partícula
                # $$v_i(t+1) = wv_i(t) + c_1r_1[\hat{x}_i(t) - x_i(t)] + c_2r_2[g(t) - x_i(t)]$$
                #calcular las velocídades
                
                for k in range(dim):
                    r1 = rnd.random()
                    r2 = rnd.random()

                    swarm[i].velocity[k] = (w*swarm[i].velocity[k] + c1*r1*swarm[i].best_part_pos[k] +
                                            c2*r2*(best_swarm_pos[k] - swarm[i].position[k]))
                    
                    #si la velocidad no está en los límites [minx,maxx]
                    if swarm[i].velocity[k]<minx:
                        swarm[i].velocity[k] = minx
                    if swarm[i].velocity[k]>maxx:
                        swarm[i].velocity[k] = maxx
                        
                #calcular la nueva posición
                for k in range(dim):
                    swarm[i].position[k] += swarm[i].velocity[k] 

                #calculando la evaluación de la posición en la función de costo
                swarm[i].fitness = fitness(swarm[i].position)

                # actualizando la mejor posición de la partícula
                if swarm[i].fitness < swarm[i].best_part_fitnessVal:
                    swarm[i].best_part_fitnessVal = swarm[i].fitness
                    swarm[i].best_part_pos = copy.copy(swarm[i].position)

                # actualizando la mejor posición de todo el enjambre
                if swarm[i].fitness < best_swarm_fitnessVal:
                    best_swarm_fitnessVal = swarm[i].fitness
                    best_swarm_pos = copy.copy(swarm[i].position)
                
            Iter+=1
        return best_swarm_pos, best_swarm_pos_hist, best_swarm_fitnessVal, best_swarm_fitnessVal_hist
    
class AG:
    
    def rulette_wheel_selection(p):
        c=np.cumsum(p)
        r=sum(p)*np.random.rand()
        
        ind= np.argwhere(r<=c)
        return ind[0][0]
    
    def crossover(p1,p2):
        c1=copy.deepcopy(p1)
        c2=copy.deepcopy(p2)
        alpha = np.random.uniform(0,1,(c1['position'].shape))
        c1['position']= alpha*p1['position'] + (1-alpha)*p2['position']
        c2['position']= alpha*p2['position'] + (1-alpha)*p1['position']
        return c1, c2
    
    def mutate(c, mu, sigma):
        y = copy.deepcopy(c)
        flag = np.random.rand(*(c['position'].shape)) <= mu
        ind = np.argwhere(flag)
        y['position'][ind] += sigma*np.random.rand(*ind.shape)
        return y
    
    def bounds(c,varmin,varmax):
        c['position'] = np.maximum(c['position'], varmin)
        c['position'] = np.minimum(c['position'], varmax)
        
    def sort(arr):
        n=len(arr)
        for i in range(n-1):
            for j in range(0,n-i-1):
                if arr[j]['cost'] > arr[j+1]['cost']:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr
    
    def min_sq_N(beta, x_points, y_points,N):
        n=len(x_points)
        f_hat_N = np.array([beta[i]*x_points**i for i in range(N+1)]).sum(axis=0)
        #f_hat = beta[0] + beta[1]*x_points + beta[2]*x_points**2 + beta[3]*x_points**3
        sqe_n = (1/(2*n))*((y_points - f_hat_N)**2).sum()
        return sqe_n

    
    def ga(self, costfun, x_points, y_points,N,num_var, varmin, varmax, maxit, npop, num_children, mu, sigma, beta):
    
        #inicializar la población
        population={}
        for i in range(npop):
            population[i] = {'position': None, 'cost':None} 
        
        bestsol = copy.deepcopy(population)
        bestsol_cost = np.inf
            
        for i in range(npop):
            population[i]['position'] = np.random.uniform(varmin, varmax, num_var)
            population[i]['cost'] = costfun(population[i]['position'],x_points, y_points,N)
            
            if population[i]['cost'] < bestsol_cost:
                bestsol = copy.deepcopy(population[i])
        
        print('best_sol: {}'.format(bestsol))
        
        bestcost = np.empty(maxit)
        bestsolution = np.empty((maxit,num_var))
        
        for it in range(maxit):
            #calcular las probabilidades de la ruleta
            costs=[]
            for i in range(len(population)):
                costs.append(population[i]['cost'])
            costs = np.array(costs)
            avg_cost = np.mean(costs)
    
            if avg_cost !=0:
                costs = costs/avg_cost
    
            props = np.exp(-beta*costs)
            
            for _ in range(num_children//2):
                
                # selección por ruleta
                p1 = population[self.rulette_wheel_selection(props)]
                p2 = population[self.rulette_wheel_selection(props)]
                
                # Crossover de los padres
                c1, c2 = self.crossover(p1,p2)
                
                # Realizar la mutación
                c1=self.mutate(c1,mu,sigma)
                c2=self.mutate(c2,mu,sigma)
                
                self.bounds(c1, varmin, varmax)
                self.bounds(c2, varmin, varmax)
                
                #evaluar la función de costo
                c1['cost'] = costfun(c1['position'],x_points, y_points,N)
                c2['cost'] = costfun(c2['position'],x_points, y_points,N)
                
                if type(bestsol_cost)==float:
                    if c1['cost']<bestsol_cost:
                        bestsol_cost = copy.deepcopy(c1)
                else:
                    if c1['cost']<bestsol_cost['cost']:
                        bestsol_cost = copy.deepcopy(c1)
    
                if c2['cost']<bestsol_cost['cost']:
                    bestsol_cost = copy.deepcopy(c2)
                    
            #juntar la poblacion de la generación anterior con la nueva
            population[len(population)] = c1
            population[len(population)] = c2
            
            population = self.sort(population)
            
            #almacenar el history
            bestcost[it] = bestsol_cost['cost']
            bestsolution[it] = bestsol_cost['position']
            
            print('iteración {}, best_sol {}, best_cost {}'.format(it,bestsolution[it],bestcost[it]))
            
        out = population
        return (out, bestsolution, bestcost)
