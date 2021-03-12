#!/bin/bash

horovodrun -np 4 -H localhost:4 python3 train_mem_model.py
#horovodrun -np 8 -H server1:4,server2:4 python3 train_mem_model.py
