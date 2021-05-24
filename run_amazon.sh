#!/bin/bash

dataset=amazon
a=0.5
b=0.5
c=0.0
d=0.0
t=0.6
rel_num=2

python3 src/train.py --dataset $dataset --prefix full_model --a $a --b $b --c $c --d $d --t $t --sample_mode 'n|l'

