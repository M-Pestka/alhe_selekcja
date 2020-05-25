#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:43:20 2020

@author: kristofer
"""

import numpy as np

#genotyp = [0,1,0,1,0,1,0,0,1,0,0,0,1,1,0,1]

g = np.array(genotyp)

zero_flip_mask = ((g==0)*np.random.uniform(size = g.shape) > (1-1e-5)).astype(int)
g = g*(1-zero_flip_mask) + (1-g)*zero_flip_mask

ones_flip_mask = ((g==1)*np.random.uniform(size = g.shape) > (1-0.01)).astype(int)
g = g*(1-ones_flip_mask) + (1-g)*ones_flip_mask

genotyp = g.tolist()