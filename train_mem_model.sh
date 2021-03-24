#!/bin/bash

#mpirun -np 4 --allow-run-as-root python3 train_mem_model.py
mpirun -np 1 --allow-run-as-root python3 train_mem_model.py
 
#horovodrun -np 8 -H server1:4,server2:4 python3 train_mem_model.py
