#!/bin/zsh
         
export PATH=/cal/softs/anaconda/anaconda3/bin:$PATH 
source activate cyclegan
python /tsi/clusterhome/glabarbera/acgan/main.py --testing TRUE
