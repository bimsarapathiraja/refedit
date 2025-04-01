#!/bin/bash

cd RefEdit/data/mask_generation

git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git

# copy files inside 'mask_generation' to 'Grounded-Segment-Anything'
cp -r mask_generation/* Grounded-Segment-Anything/

cd RefEdit/metrics/pnpmetrics

git clone https://github.com/cure-lab/PnPInversion.git

# the evaluation folder already exists in the 'PnPInversion' folder. add everything in evaluation to the pnpinversion/evaluation folder
cp -r evaluation/* PnPInversion/evaluation/

# run_eval folder doesnt exist in the 'PnPInversion' folder. add everything in run_eval to the pnpinversion/run_eval folder
mkdir PnPInversion/run_eval
cp -r run_eval/* PnPInversion/run_eval/

cd RefEdit/training/RefEdit-SD1.5
git clone https://github.com/timothybrooks/instruct-pix2pix.git

# three files and one folder are inside refedit1.5. The folder 'configs' is already in the instruct-pix2pix folder. add everything in configs to the instruct-pix2pix/configs folder. But other files are not in the instruct-pix2pix folder. So add them to the instruct-pix2pix folder.
cp -r configs/* instruct-pix2pix/configs/
cp -r scripts/* instruct-pix2pix/

cd RefEdit/training/RefEdit-SD3
git clone https://github.com/pkunlp-icler/UltraEdit.git

cp -r scripts/* UltraEdit/scripts/
cp -r training/* UltraEdit/traning/

echo "Setup complete"