#!/bin/bash

for instruction in "turn the biggest mushroom color to yellow" "remove the biggest mushroom" "remove all mushrooms"
do
    for ckpt in "checkpoints/instruct-pix2pix-00-22000.ckpt" "checkpoints/MagicBrush-epoch-52-step-4999.ckpt" "checkpoints/MagicBrush-epoch-000168.ckpt" "/home/cr8dl-user/DATA_bim/training/finetune_combined_ip2p_2/checkpoints/trainstep_checkpoints/epoch=000024-step=000020999.ckpt"
    do
        # Create a unique filename for each instruction and ckpt
        # Replace spaces and special characters in instruction and ckpt to make them filename-friendly
        instruction_clean=$(echo "$instruction" | tr ' ' '_' | tr '/' '_')
        ckpt_clean=$(basename "$ckpt" | tr ' ' '_' | tr '/' '_' | sed 's/\.ckpt//')

        # Define the output file path
        output_file="outputs/mushroom/${instruction_clean}_${ckpt_clean}.jpg"

        # Print the command being executed (for debugging)
        echo "Running: python edit_cli.py --input imgs/mushroom.jpg --output $output_file --edit '$instruction' --ckpt '$ckpt'"

        # Execute the command
        python edit_cli.py --input imgs/mushroom.jpg --output "$output_file" --edit "$instruction" --ckpt "$ckpt"
    done
done