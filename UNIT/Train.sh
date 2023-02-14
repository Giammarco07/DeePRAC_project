#!/bin/zsh
         
export PATH=/cal/softs/anaconda/anaconda3/bin:$PATH 
source activate unit
python /tsi/clusterhome/glabarbera/UNIT/train.py --config /tsi/clusterhome/glabarbera/UNIT/configs/unit_contrast2noncontrast_folder.yaml --trainer UNIT 
