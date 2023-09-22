#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import math
from typing import List, Tuple, Set, FrozenSet, Dict, Iterable, Iterator
from typing import Any, Optional, Sequence, Mapping
from collections import Counter
import numpy as np
import sympy as sp
from sklearn.preprocessing import normalize
import time

DSM = np.array


# ### Read .etf File and Compile Gantt Data

# In[2]:


start_time = time.time()


# In[3]:


FOLDER_NAME = 'DIRECTORY'
FILE_NAME = 'FILE'

EVALUATE_CLAIMS = False #If TRUE claims will be used instead of resources for DSM rows/columns.

resources = []             # <resource id> <attribute>
resource_dependencies = [] # <dep id>      <type>    <start resource id>    <end resource id>
rd_len = []                # <horizontal length>

claims = []                # <claim id>    <t0>      <t1>                   <resource id>
claim_dependencies = []    # <dep id>      <type>    <start claim id>       <end claim id>
cd_len = []                # <horizontal length>

attributes_raw = []

with open(f'{FOLDER_NAME}/{FILE_NAME}.etf', 'r') as file:
    # Save all lines starting with R, D, or D (resource, claim, or dependency code)
    i = 1
    for line in file:        
        if line[0] == 'R': # Decompose resource data
            resource = [int(line.split(';')[0].split()[1])]
            if not EVALUATE_CLAIMS: 
                attribute_raw = line.split(';')[1].split(',')
                attribute_labels = {i.split('=')[0] for i in attribute_raw}
                attribute = [i.split('=') for i in attribute_raw if i.split('=')[0] == list(attribute_labels)[0]][0]
                resource.append(attribute[1][:-1])
            resources.append(resource)
        
        elif line[0] == 'C': # Decompose claim data
            if EVALUATE_CLAIMS: attributes_raw.append(line.split(';')[1].split(','))
            
            claim = line.split(';')[0].split()
            c_append = []
            
            for i in range(1,5):
                if '.' not in claim[i]: #Checks if int(claim[i]) is valid
                    c_append.append(int(claim[i]))
                else:
                    c_append.append(float(claim[i]))
            
            claims.append(tuple(c_append))
        
        elif line[0] == 'D': # Decompose dependency data
            dep = line.split(';')[0].split()
            claim_dependencies.append(tuple(int(dep[i]) for i in range(1,5)))
            
#Sort resource and claim lists by ID
resources.sort(key = lambda x: x[0])
claims.sort(key = lambda x: x[0])
       
#Add horizontal length to claim dependencies
for d in range(len(claim_dependencies)):
    t0, t1 = claims[claim_dependencies[d][2]][2], claims[claim_dependencies[d][3]][1]
    cd_len.append(t1-t0)
    
#Generate list of inter-resource dependencies
for d in range(len(claim_dependencies)):
    dep = list(claim_dependencies[d])    
    dep[2] = claims[dep[2]][3]
    dep[3] = claims[dep[3]][3]
    
    if dep[2] != dep[3]:
        resource_dependencies.append(tuple(dep))
        rd_len.append(cd_len[d])
        
# print('R: ', resources)
# print('\nC: ', claims)
# print('\nC_D: ', claim_dependencies)
# print('\nR_D: ', resource_dependencies)
# print('\nCD_len: ', cd_len)
# print('\nRD_len: ', rd_len)

gantt_data = {}

if EVALUATE_CLAIMS:
    for claim in claims:
        key = claim[0]
        end_claim = [cd[3] for cd in claim_dependencies if cd[2] == key]
        _len = [cd_len[i] for i in range(len(claim_dependencies)) if claim_dependencies[i][2] == key]
        _att = attributes[key][1][:-1]
        if key in gantt_data.keys():
            gantt_data[key]["con"] + end_claim            
            gantt_data[key]["len"] + _len
        else:
            gantt_data[key] = {"con": [], "len": [], "att": str}
            gantt_data[key]["con"] = end_claim
            gantt_data[key]["len"] = _len
            gantt_data[key]["att"] = _att
else:
    for i in range(len(resources)):
        key = resources[i][0]
        gantt_data[key] = {"con": [], "len": [], "att": str}
        gantt_data[key]["con"] = [rd[3] for rd in resource_dependencies if rd[2] == key]
        gantt_data[key]["len"] = [rd_len[i] for i in range(len(resource_dependencies)) if resource_dependencies[i][2] == key]
        gantt_data[key]["att"] = resources[i][1]
    
print(gantt_data)


# ### Generate DSM from Gantt Data

# In[4]:


global n_resources
n_resources = len(gantt_data.keys())

DSM_raw: DSM = np.zeros((n_resources, n_resources), list)
    
for row in list(gantt_data.keys()):
    if gantt_data[row]["con"]:
        _row = gantt_data[row]["con"]
        _len = gantt_data[row]["len"]
        for col in range(len(_row)):
            if DSM_raw[row, _row[col]] == 0:
                x, y = [], abs(row-_row[col])
                x.append(gantt_data[row]["len"][col])
                DSM_raw[row, _row[col]] = [x, y] # <x len> <y len>
            else:
                DSM_raw[row, _row[col]][0].append(gantt_data[row]["len"][col])             
                        
print(DSM_raw)


# ### Cost Function

# In[5]:


def TotalOrderCost(_dsm: DSM) -> float:
    """Calculate the total cost of a solution based on lane order. This is defined as the sum of the product
    of interaction length and count in _dsm. 
    
    The tuple values in _dsm represent horizontal length, vertical length, and number of interactions 
    between a row and column.
    """
    cost = sum([sum([sum([math.sqrt((i**2)+(col[1]**2)) for i in col[0]]) for col in row if col != 0]) for row in _dsm])
    return cost


# ### Genetic Algorithm

# In[6]:


def ReOrderDSM(_dsm: DSM, order: list) -> DSM:
        """Re-orders the rows and columns of _dsm according to order. Returns the re-ordered DSM.
        """
        _dsm_new = np.asarray([_dsm[i] for i in order])
        _dsm_new = _dsm_new.transpose()
        _dsm_new = np.asarray([_dsm_new[i] for i in order])
        _dsm_new = _dsm_new.transpose()

        for row in range(len(_dsm_new)):
            for col in range(len(_dsm_new[row])):
                if _dsm_new[row][col] != 0:
                    _dsm_new[row][col][1] = abs(row-col)
        
        return _dsm_new


# In[7]:


def GA(_dsm: DSM, max_iter: int, population: int, moves: int = 1) -> list:
    """Genetic algorithm.
    
    Solutions are represented as a vector of size equal to the resources with values pertaining to the resource number
    and vector position pertaining to the order of the resources.
    
    Population count is chosen arbitrarily and is equal to the number of solutions.
    """
    
    DEFAULT_SOLUTION = [i for i in range(n_resources)]
    solution_data = [random.sample(DEFAULT_SOLUTION, n_resources) for i in range(population)]
    cost_data = [TotalOrderCost(ReOrderDSM(_dsm, sol)) for sol in solution_data]
    
    best_solution = (solution_data[cost_data.index(min(cost_data))], min(cost_data))
    
#     print(solution_data, end='\n\n')
#     print(cost_data, end='\n\n')
#     print(best_solution)
    
    for i in range(max_iter): #Itterate
#         print(f"\nItteration {i} ---------------------------------------------------------------")
        
        new_solutions = [best_solution[0] for i in range(population)]
                
        for j in range(population): #Itterate over each solution
#             print(f"Player {j}:")
            
            active_solution = new_solutions[j].copy()
                        
            for m in range(moves): #Perform moves on the solution
                target_resource = np.random.choice([key for key in gantt_data.keys() if gantt_data[key]["con"]])
                connected_resource = np.random.choice(gantt_data[target_resource]["con"])
                neighbour_side = np.random.choice([-1,1])
                
                if target_resource + neighbour_side < 0:
                    neighbour_resource = target_resource + 1
                elif target_resource + neighbour_side >= n_resources:
                    neighbour_resource = target_resource - 1
                else:
                    neighbour_resource = target_resource + neighbour_side
                
#                 print(target_resource, connected_resource, neighbour_resource)
                if connected_resource != neighbour_resource:
                    get = (active_solution.index(connected_resource), active_solution.index(neighbour_resource))

                    active_solution[get[0]] = neighbour_resource
                    active_solution[get[1]] = connected_resource
                
#                 print(active_solution)
                
            new_solutions[j] = active_solution
        
        solution_data = new_solutions
        cost_data = [TotalOrderCost(ReOrderDSM(_dsm, sol)) for sol in solution_data]
        current_best_solution = (solution_data[cost_data.index(min(cost_data))], min(cost_data))
        
        if current_best_solution[1] > best_solution[1]:
            best_solution = current_best_solution
            
    return(best_solution[0])


# ### Apply Solution to DSM

# In[8]:


sorting_t0 = time.time()

new_order = GA(DSM_raw, 200, 200, 1)

sorting_t1 = time.time()

print('New Resource Order:', new_order)
print('Sorting Time:', round(sorting_t1-sorting_t0, 5))


# ### Evalute Solution Against Manual Resource Order

# In[9]:


manual_order = sorted(list(gantt_data.keys()), key = lambda k: gantt_data[k]['att'])
print('Manual Resource Order:', manual_order)

if TotalOrderCost(ReOrderDSM(DSM_raw, manual_order)) < TotalOrderCost(ReOrderDSM(DSM_raw, new_order)):
    new_order = manual_order
    print('New Resource Order:', new_order)

print('Original Cost:', TotalOrderCost(ReOrderDSM(DSM_raw, manual_order)))
print('Final Cost:', TotalOrderCost(ReOrderDSM(DSM_raw, new_order)))


# ### Modify .etf File

# In[10]:


OUTPUT_SUFFIX = '_GA_sorted'
GRP_NAME = 'claim_order' if EVALUATE_CLAIMS else 'A_Order'
F_CLAIM_GROUPING = ',{}={}\n'
F_RESOURCE_GROUPING = '{}={},'

with open(f'{FOLDER_NAME}/{FILE_NAME}.etf', 'r') as file:
    lines = [line for line in file]

resource_indices = [i for i in range(len(lines)) if lines[i][0] == 'R']
resource_indices.sort(key = lambda i: int(lines[i].split()[1]))
claim_indices = [i for i in range(len(lines)) if lines[i][0] == 'C']

if EVALUATE_CLAIMS:
    for c_i in range(n_resources):
        order_str = str(c_i).zfill(n_resources // 10)
        new_line = lines[claim_indices[new_order[c_i]]][:-1]+F_GROUPING.format(GRP_NAME,order_str)
        lines[claim_indices[new_order[c_i]]] = new_line
else:
    for r_i in range(n_resources):
        order_str = str(r_i).zfill(n_resources // 10)
        line_segment = lines[resource_indices[new_order[r_i]]].split(';')
        line_segment[1] = F_RESOURCE_GROUPING.format(GRP_NAME,order_str) + line_segment[1]
        new_line = ';'.join(line_segment)
        lines[resource_indices[new_order[r_i]]] = new_line

with open(f'{FOLDER_NAME}/Results/{FILE_NAME}{OUTPUT_SUFFIX}.etf', 'w') as f_out:
    [f_out.write(line) for line in lines]


# In[11]:


end_time = time.time()

print(f'Processing time was {round(end_time - start_time, 5)}s')

