import os
import glob 
import numpy as np
from typing import List
import functools
from itertools import *
import pprint

pp = pprint.PrettyPrinter(indent=4)

print(pwd)

rootpath = "../data/Ex1/TrainSet"

# data definition
labelList = []
filePathList = []
nb_class = None
dataContainer = None
pair_same = None
pair_diff = None
ClassPair = None

for root , subs , files in os.walk(rootpath):
    if len(files) == 0:
        continue # 여기서 클래스 종류별 맵핑이 가능하다. 
    for name in files:
        filepath = os.path.join(root,name)
        label = root.split('\\')[-1]
        labelList.append(label)
        filePathList.append(filepath)
        
        
# split eeach class data
nb_class = list(set(labelList))
nb_class = list(map(lambda x : int(x) , nb_class))
dataContainer = {}
for i_class in nb_class:
    filtered = list(filter(lambda x : int(x[0]) == i_class , zip(labelList, filePathList)))
    dataContainer[i_class] = [x[1] for x in filtered]
    
# Start Create Same Class Data ,  nC2

pair_same = {}

for k,v in dataContainer.items():
    pair_same[k] = list(combinations(v,2))
    pp.pprint(len(pair_same[k]))
    
# start Creaete Not Same Class Data
pair_diff = {}
ClassPair = list( combinations( nb_class , 2) )

for combi in ClassPair:
    pair_diff[combi] = list(product(dataContainer[1],dataContainer[combi[1]]))

print()
