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

from itertools import permutations

DSM = np.array


# ### Read .etf File and Compile Gantt Data

# In[ ]:


start_time = time.time()


# In[2]:


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

# In[3]:


global n_resources
n_resources = len(gantt_data.keys())

DSM_raw: DSM = np.zeros((n_resources, n_resources), list)
DSM_raw_merged: DSM = np.zeros((n_resources, n_resources), float)
            
for row in list(gantt_data.keys()):
    if gantt_data[row]["con"]:
        _row = gantt_data[row]["con"]
        _len = gantt_data[row]["len"]
        count = Counter(_row)
        for col in range(len(_row)):
            DSM_raw_merged[row, _row[col]] = 1/count[_row[col]]
            if DSM_raw[row, _row[col]] == 0:
                x, y = [], abs(row-_row[col])
                x.append(gantt_data[row]["len"][col])
                DSM_raw[row, _row[col]] = [x, y] # <x len> <y len>
            else:
                DSM_raw[row, _row[col]][0].append(gantt_data[row]["len"][col])    
            
print(DSM_raw)
print(DSM_raw_merged)


# ### Markov Clustering

# In[4]:


def MCL(_dsm: DSM, evaporation_const: float, expansion_const: float, inflation_const: float) -> list:
    """Markov clustering method for DSM. Based on Wilschut2018
    
    Solutions are represented as a vector of size equal to the resources with values pertaining to the cluster
    each resource is part of.
    """
    
    M_evap = evaporation_const * np.identity(n_resources) 
    _dsm_transpose = _dsm.transpose()
    
    f_in_lst = [] #Storage for inflow vectors
    df_lst = [] #Storage for flow rate vectors, to be used when concatenating to form flow rate matrix F
    df_rev_lst = [] #Storage for flow rate vectors for reversed flow. Used to form matrix F_rev
    
    for i in range(n_resources):
        # Define input vector
        f_in = np.array([0 for n in range(n_resources)]).reshape((n_resources,1))
        f_in[i] = 1
        f_in_lst.append(f_in)
        
        # Define sink/source matrices
        W_in = np.zeros((n_resources,n_resources))
        W_out = np.zeros((n_resources,n_resources))
        
        sum_w_in = sum([_dsm[i][j] for j in range(n_resources)])
                
        for j in range(n_resources):
            sum_w_out = sum([_dsm[i][j] for i in range(n_resources)])
            W_in[j][j] = max(1, sum_w_in)
            W_out[j][j] = max(1, sum_w_out)
            
        f = np.linalg.solve(np.dot(_dsm, np.linalg.inv(np.dot(W_out, M_evap))) - W_in , -f_in)
        f_rev = np.linalg.solve(np.dot(_dsm_transpose, np.linalg.inv(np.dot(W_in, M_evap))) - W_out , -f_in)
        
        df_lst.append(f)
        df_rev_lst.append(f_rev)
    
    F_in = np.concatenate(tuple(f_in_lst), axis=1) #Inflow matrix
    F = np.concatenate(tuple(df_lst), axis=1) #Flow rate matrix
    F_rev = np.concatenate(tuple(df_rev_lst), axis=1) #Flow rate matrix for reversed flow
    
    M_influence = F - np.dot(np.linalg.inv(W_in),F_in) #Matrix of influences between resources
    M_dependency = F_rev - np.dot(np.linalg.inv(W_out),F_in) #Matrix of dependencies between resources

    Q = M_influence + M_dependency
    
    print(F_in, end='\n\n')
    print(F, end='\n\n')
    print(F_rev, end='\n\n--------------------------------------------------------------\n\n')
    print(M_influence, end='\n\n')
    print(M_dependency, end='\n\n')
    print(Q, end='\n\n')

    cp = 1 / (evaporation_const * max([(W_in+W_out)[i][i] for i in range(n_resources)])**(expansion_const+1))
    
    for i in range(len(Q)):
        for j in range(len(Q)):
            if Q[i][j] < cp:
                Q[i][j] = 0.0
    
    P = normalize(Q, axis = 0, norm = 'l1') #Transition probability matrix
    print(P, end='\n\n')
    
    while True:
        P1 = np.linalg.matrix_power(P,expansion_const) #Expansion
        P1 = np.power(P1, inflation_const)#Inflation
        P1 = normalize(P1, axis = 0, norm = 'l1')        
        if np.linalg.norm((P1-P), ord=2) < 0.005:
            P = P1
            break
        else:
            P = P1
    
    P = np.asarray(np.rint(P), dtype=int)
    print(P)
    
    #Convert Matrix P into solution vector
    solution = []   
    P_t = P.transpose()
    solo_cluster = n_resources+1
    for resource in range(n_resources):
        if max(P_t[resource]) == 0:
            solution.append(solo_cluster)
            solo_cluster += 1
        else:
            for cluster in range(n_resources):
                if P_t[resource][cluster] == 1:
                    solution.append(cluster+1)
    
    return solution 


# ### Apply Solution to DSM

# In[5]:


def TotalOrderCost(_dsm: DSM) -> float:
    """Calculate the total cost of a solution based on lane order. This is defined as the sum of the product
    of interaction length and count in _dsm. 
    
    The tuple values in _dsm represent horizontal length, vertical length, and number of interactions 
    between a row and column.
    """
    cost = sum([sum([sum([math.sqrt((i**2)+(col[1]**2)) for i in col[0]]) for col in row if col != 0]) for row in _dsm])
    return cost


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


def ClusterToOrder(clusters: dict, key_order: tuple) -> List[int]:
    """Converts a cluster dictionary into a 1D element order list which can be applied to a DSM.
    """
    new_order = []
    [[new_order.append(v) for v in clusters[k]] for k in key_order]
    return new_order


# In[8]:


sorting_t0 = time.time()

solution = MCL(DSM_raw_merged, 1.5, 2, 1.6) # evap, expa, infl

cluster_dict = {}

for i in range(len(solution)):
    key = solution[i]
    if key in cluster_dict.keys():
        cluster_dict[key].append(i)
    else:
        cluster_dict[key] = []
        cluster_dict[key].append(i)

print('Solution Vector:', solution)
print('Clusters:', cluster_dict)

default_cluster_order = list(cluster_dict.keys())
default_resource_order = ClusterToOrder(cluster_dict, default_cluster_order)
_permutations = permutations(default_cluster_order)

best_cluster_order = (default_cluster_order, TotalOrderCost(ReOrderDSM(DSM_raw, default_resource_order)))
for i in range(math.factorial(len(default_cluster_order))):
    key_order = next(_permutations)
    resource_order = ClusterToOrder(cluster_dict, key_order)
    cost = TotalOrderCost(ReOrderDSM(DSM_raw, resource_order))

    if cost < best_cluster_order[1]:
        best_cluster_order = (key_order, cost)
    
ordered_dict = {k : cluster_dict[k] for k in best_cluster_order[0]}
print('Optimal Cluster Order:', ordered_dict)

new_order = ClusterToOrder(ordered_dict, ordered_dict.keys())

final_dict = ordered_dict.copy()
for k in ordered_dict.keys():
    if len(ordered_dict[k]) > 1:
        default_sub_order = ordered_dict[k]
        _permutations = permutations(default_sub_order)

        _best = (default_sub_order, TotalOrderCost(ReOrderDSM(DSM_raw, new_order)))
        _dict = ordered_dict.copy()
        for i in range(math.factorial(len(default_sub_order))):
            perm = next(_permutations)
            _dict[k] = perm        
            cost = TotalOrderCost(ReOrderDSM(DSM_raw, ClusterToOrder(_dict, _dict.keys())))
            if cost < _best[1]:
                _best = (perm, cost)

        final_dict[k] = list(_best[0])

new_order = ClusterToOrder(final_dict, final_dict.keys())
        
print('New Resource Order:', new_order)

sorting_t1 = time.time()
print('Sorting Time:', round(sorting_t1-sorting_t0, 5))


# ### Evalute Solution Against Manual Resource Order

# In[ ]:


manual_order = sorted(list(gantt_data.keys()), key = lambda k: gantt_data[k]['att'])
print('Manual Resource Order:', manual_order)

if TotalOrderCost(ReOrderDSM(DSM_raw, manual_order)) < TotalOrderCost(ReOrderDSM(DSM_raw, new_order)):
    new_order = manual_order
    print('New Resource Order:', new_order)
    
print('Original Cost:', TotalOrderCost(ReOrderDSM(DSM_raw, manual_order)))
print('Final Cost:', TotalOrderCost(ReOrderDSM(DSM_raw, new_order)))


# ### Modify .etf File

# In[ ]:


OUTPUT_SUFFIX = '_MCL_sorted'
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


# In[ ]:


end_time = time.time()

print(f'Processing time was {round(end_time - start_time, 5)}s')

