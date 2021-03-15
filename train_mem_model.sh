#!/bin/bash

# should specify number of GPU, refer to https://horovod.readthedocs.io/en/stable/running_include.html
horovodrun -np 1 -H localhost:1 python3 train_mem_model.py
#horovodrun -np 4 -H localhost:4 python3 train_mem_model.py
#horovodrun -np 8 -H server1:4,server2:4 python3 train_mem_model.py
