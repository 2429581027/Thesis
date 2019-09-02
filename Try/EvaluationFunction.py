#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:58:17 2019

@author: chaotang
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

# define the chamfer distance
def chamfer_dist(clean_pc,noisy_pc):
    # Get How many points in each point set
    noisy_num = noisy_pc.shape[0]
    clean_num = clean_pc.shape[0]
    # Use KNN to find the nearest neighbours for each point
    clean_Neighbour = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(clean_pc)
    Noisy_Neighbour = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(noisy_pc)
    
    distances_to_clean, _ = clean_Neighbour.kneighbors(noisy_pc)   
    # every point in noisy_pc 's close distance to clean point cloud
    # print('distances_to_clean',distances_to_clean.shape)
    
    distances_to_noisy, _ = Noisy_Neighbour.kneighbors(clean_pc)
    # print('distances_to_Noisy',distances_to_noisy.shape)
    
    term1 = np.reshape(distances_to_clean,(-1,))
    term2 = np.reshape(distances_to_noisy,(-1,))
    
    term1 = sum(np.square(term1)) / noisy_num
    term2 = sum(np.square(term2)) / clean_num

    total_distance = term1 + term2
    
    return total_distance

def RMSD(clean_pc,noisy_pc):# root mean square distance-to-surface
    # Get How many points in each point set
    noisy_num = noisy_pc.shape[0]
    # Use KNN to find the nearest neighbours for each point
    clean_Neighbour = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(clean_pc)
    distances_to_clean, _ = clean_Neighbour.kneighbors(noisy_pc)
    
    term1 = np.reshape(distances_to_clean,(-1,))

    term1 = sum(np.square(term1)) / noisy_num
    
    dist = np.sqrt(term1)
    return dist

if __name__=='__main__':
    clean_pc = np.random.normal(size = (100,3))
    noisy_pc = np.random.normal(size = (500,3))
    print(chamfer_dist(clean_pc,noisy_pc))
    print(RMSD(clean_pc,noisy_pc))