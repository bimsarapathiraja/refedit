#!/bin/bash

cd data/mask_generation
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cp -r scripts/* Grounded-Segment-Anything/  

cd ../../metrics/pnpmetrics
git clone https://github.com/cure-lab/PnPInversion.git
cp -r evaluation/* PnPInversion/evaluation/
mkdir PnPInversion/run_eval
cp -r run_eval/* PnPInversion/run_eval/
cp run_editing_ip2p_refedit_final.py PnPInversion/

cd ../../training/RefEdit-SD1.5
git clone https://github.com/timothybrooks/instruct-pix2pix.git
cp -r configs/* instruct-pix2pix/configs/
cp -r scripts/* instruct-pix2pix/

cd ../../training/RefEdit-SD3
git clone https://github.com/pkunlp-icler/UltraEdit.git
cp -r scripts/* UltraEdit/scripts/
cp -r training/* UltraEdit/traning/

echo "Setup complete"