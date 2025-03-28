# conda deactivate
# deactivate
# source activate ip2p2

for ckpt in CKPT_PATH 
do
    CKPT="DIR_PATH/$ckpt"
    OUTPUT_PATH="SAVE_PTH/$ckpt/refedit/results"

    echo "************************************"
    echo "Running editing for $CKPT"
    echo "Running editing for $OUTPUT_PATH"
    echo "************************************"

    python run_editing_ip2p_refedit_final.py --checkpoint $CKPT --output_path $OUTPUT_PATH --device "cuda:0"

    python evaluation/evaluate_refedit_final.py \
        --metrics "structure_distance" "psnr_unedit_part" "lpips_unedit_part" "mse_unedit_part" "ssim_unedit_part" "clip_similarity_source_image" "clip_similarity_target_image" "clip_similarity_target_image_edit_part" \
        --result_path $OUTPUT_PATH3/evaluation_result_final_512.csv \
        --edit_category_list 0 1 2 3 4 5 6 7 8 9 \
        --tgt_methods $OUTPUT_PATH3
done
