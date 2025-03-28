import json
import argparse
import os
import numpy as np
from PIL import Image
import csv
from matrics_calculator import MetricsCalculator
from torchvision import transforms

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array



def calculate_metric(metrics_calculator,metric, src_image, tgt_image, src_mask, tgt_mask,src_prompt,tgt_prompt):
    try:
        if metric=="psnr":
            return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
        if metric=="lpips":
            return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
        if metric=="mse":
            return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
        if metric=="ssim":
            return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
        if metric=="structure_distance":
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
        if metric=="psnr_unedit_part":
            if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
                return "nan"
            else:
                return metrics_calculator.calculate_psnr(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
        if metric=="lpips_unedit_part":
            if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
                return "nan"
            else:
                return metrics_calculator.calculate_lpips(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
        if metric=="mse_unedit_part":
            if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
                return "nan"
            else:
                return metrics_calculator.calculate_mse(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
        if metric=="ssim_unedit_part":
            if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
                return "nan"
            else:
                return metrics_calculator.calculate_ssim(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
        if metric=="structure_distance_unedit_part":
            if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
                return "nan"
            else:
                return metrics_calculator.calculate_structure_distance(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
        if metric=="psnr_edit_part":
            if src_mask.sum()==0 or tgt_mask.sum()==0:
                return "nan"
            else:
                return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
        if metric=="lpips_edit_part":
            if src_mask.sum()==0 or tgt_mask.sum()==0:
                return "nan"
            else:
                return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
        if metric=="mse_edit_part":
            if src_mask.sum()==0 or tgt_mask.sum()==0:
                return "nan"
            else:
                return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
        if metric=="ssim_edit_part":
            if src_mask.sum()==0 or tgt_mask.sum()==0:
                return "nan"
            else:
                return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
        if metric=="structure_distance_edit_part":
            if src_mask.sum()==0 or tgt_mask.sum()==0:
                return "nan"
            else:
                return metrics_calculator.calculate_structure_distance(src_image, tgt_image, src_mask, tgt_mask)
        if metric=="clip_similarity_source_image":
            return metrics_calculator.calculate_clip_similarity(src_image, src_prompt,None)
        if metric=="clip_similarity_target_image":
            return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt,None)
        if metric=="clip_similarity_target_image_edit_part":
            if tgt_mask.sum()==0:
                return "nan"
            else:
                return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt,tgt_mask)
    except:
        return "nan"
    
all_tgt_image_folders={
    # results of comparing inversion
    # ---
    "1_ddim+p2p":"output/ddim+p2p/annotation_images",
    "1_null-text-inversion+p2p_a800":"output/null-text-inversion+p2p_a800/annotation_images",
    "1_null-text-inversion+p2p_3090":"output/null-text-inversion+p2p_3090/annotation_images",
    "1_negative-prompt-inversion+p2p":"output/negative-prompt-inversion+p2p/annotation_images",
    "1_stylediffusion+p2p":"output/stylediffusion+p2p/annotation_images",
    "1_directinversion+p2p":"output/directinversion+p2p/annotation_images",
    # ---
    "1_ddim+masactrl":"output/ddim+masactrl/annotation_images",
    "1_directinversion+masactrl":"output/directinversion+masactrl/annotation_images",
    # ---
    "1_ddim+pix2pix-zero":"output/ddim+pix2pix-zero/annotation_images",
    "1_directinversion+pix2pix-zero":"output/directinversion+pix2pix-zero/annotation_images",
    # ---
    "1_ddim+pnp":"output/ddim+pnp/annotation_images",
    "1_directinversion+pnp":"output/directinversion+pnp/annotation_images",
    # ---
    # results of comparing model-based methods
    "2_instruct-pix2pix":"/scratch/nkusumba/img_edit/output/instruct-pix2pix/annotation_images",
    "2_instruct-diffusion":"/scratch/nkusumba/img_edit/output/instruct-diffusion/annotation_images",
    "2_instruct-diffusion-ha": "/scratch/nkusumba/img_edit/output/instruct-diffusion/_2/annotation_images",
    "2_blended-latent-diffusion":"/scratch/nkusumba/img_edit/output/blended-latent-diffusion/annotation_images",
    "2_blended-latent-diffusion-edit":"/scratch/nkusumba/img_edit/output/blended-latent-diffusion/editing_instruct/annotation_images",
    "2_instruct-pix2pix_1": "/scratch/nkusumba/img_edit/output/instruct-pix2pix/MagicBrush-epoch-000168/annotation_images",
    "2_instruct-pix2pix_2": "/scratch/nkusumba/img_edit/output/instruct-pix2pix/MagicBrush-epoch-52-step-4999/annotation_images",
    "2_instruct-pix2pix_ours": "/scratch/nkusumba/img_edit/output/instruct-pix2pix/stage1_ours_epoch=480step=25999/annotation_images",
    "2_directinversion+p2p":"output/directinversion+p2p/annotation_images",

    "our-epoch68": "/scratch/nkusumba/img_edit/output/our-epoch68/stage1_ours_epoch=480step=25999/annotation_images",
    "our-epoch77": "/scratch/nkusumba/img_edit/output/our-epoch77/stage1_ours_epoch=480step=25999/annotation_images",
    "our-epoch95": "/scratch/nkusumba/img_edit/output/our-epoch95/stage1_ours_epoch=480step=25999/annotation_images",


    # refedit model
    "2_instruct-pix2pix-refedit":"/scratch/nkusumba/classified_benchmark/output/instruct-pix2pix-refedit",
    "2_instruct-diffusion-refedit":"/scratch/nkusumba/classified_benchmark/output/instruct-diffusion-refedit",
    "2_instruct-diffusion-refedit-ha": "/scratch/nkusumba/classified_benchmark/output/instruct-diffusion-refedit-ha",
    "2_blended-latent-diffusion-refedit":"/scratch/nkusumba/classified_benchmark/output/blended-latent-diffusion-refdit",
    "2_magicbrush-best-refedit": "/scratch/nkusumba/classified_benchmark/output/magicbrush-best-refedit",
    "2_magicbrush-paper-refedit": "/scratch/nkusumba/classified_benchmark/output/magicbrush-paper-refedit",
    "2_ours-refedit": "/scratch/nkusumba/classified_benchmark/output/ours-refedit",

    # results of different inversion/forward guidance scale
    "3_directinversion+p2p_guidance_0_1":"output/directinversion+p2p_guidance_0_1/annotation_images",
    "3_directinversion+p2p_guidance_0_5":"output/directinversion+p2p_guidance_0_5/annotation_images",
    "3_directinversion+p2p_guidance_0_25":"output/directinversion+p2p_guidance_0_25/annotation_images",
    "3_directinversion+p2p_guidance_0_75":"output/directinversion+p2p_guidance_0_75/annotation_images",
    "3_directinversion+p2p_guidance_1_1":"output/directinversion+p2p_guidance_1_1/annotation_images",
    "3_directinversion+p2p_guidance_1_5":"output/directinversion+p2p_guidance_1_5/annotation_images",
    "3_directinversion+p2p_guidance_1_25":"output/directinversion+p2p_guidance_1_25/annotation_images",
    "3_directinversion+p2p_guidance_1_75":"output/directinversion+p2p_guidance_1_75/annotation_images",
    "3_directinversion+p2p_guidance_25_1":"output/directinversion+p2p_guidance_25_1/annotation_images",
    "3_directinversion+p2p_guidance_25_5":"output/directinversion+p2p_guidance_25_5/annotation_images",
    "3_directinversion+p2p_guidance_25_25":"output/directinversion+p2p_guidance_25_25/annotation_images",
    "3_directinversion+p2p_guidance_25_75":"output/directinversion+p2p_guidance_25_75/annotation_images",
    "3_directinversion+p2p_guidance_5_1":"output/directinversion+p2p_guidance_5_1/annotation_images",
    "3_directinversion+p2p_guidance_5_5":"output/directinversion+p2p_guidance_5_5/annotation_images",
    "3_directinversion+p2p_guidance_5_25":"output/directinversion+p2p_guidance_5_25/annotation_images",
    "3_directinversion+p2p_guidance_5_75":"output/directinversion+p2p_guidance_5_75/annotation_images",
    "3_directinversion+p2p_guidance_75_1":"output/directinversion+p2p_guidance_75_1/annotation_images",
    "3_directinversion+p2p_guidance_75_5":"output/directinversion+p2p_guidance_75_5/annotation_images",
    "3_directinversion+p2p_guidance_75_25":"output/directinversion+p2p_guidance_75_25/annotation_images",
    "3_directinversion+p2p_guidance_75_75":"output/directinversion+p2p_guidance_75_75/annotation_images",
    # results of background preservation method
    "4_null-text-inverse+p2p_a800":"output/null-text-inversion+p2p_a800/annotation_images",
    "4_null-text-inverse+p2p_3090":"output/null-text-inversion+p2p_3090/annotation_images",
    "4_null-text-inversion+proximal-guidance":"output/null-text-inversion+proximal-guidance/annotation_images",
    "4_negative-prompt-inversion+proximal-guidance":"output/negative-prompt-inversion+proximal-guidance/annotation_images",
    "4_edit-friendly-inversion+p2p":"output/edit-friendly-inversion+p2p/annotation_images",
    "4_edict+direct_forward":"output/edict+direct_forward/annotation_images",
    "4_edict+p2p":"output/edict+p2p/annotation_images",
    "4_directinversion+p2p":"output/directinversion+p2p/annotation_images",
    # ablation results of contrast null-text-inversion with directinversion
    "5_ablation_directinversion_04+p2p":"output/ablation_directinversion_04+p2p/annotation_images",
    "5_ablation_directinversion_08+p2p":"output/ablation_directinversion_08+p2p/annotation_images",
    "5_ablation_null-latent-inversion+p2p_a800":"output/ablation_null-latent-inversion+p2p_a800/annotation_images",
    "5_ablation_null-latent-inversion+p2p_3090":"output/ablation_null-latent-inversion+p2p_3090/annotation_images",
    "5_ablation_null-text-inversion_single_branch+p2p_a800":"output/ablation_null-text-inversion_single_branch+p2p_a800/annotation_images",
    "5_ablation_null-text-inversion_single_branch+p2p_3090":"output/ablation_null-text-inversion_single_branch+p2p_3090/annotation_images",
    # ablation results of different intervals
    "6_ablation_directinversion_interval_2":"output/ablation_directinversion_interval_2+p2p/annotation_images",
    "6_ablation_directinversion_interval_5":"output/ablation_directinversion_interval_5+p2p/annotation_images",
    "6_ablation_directinversion_interval_10":"output/ablation_directinversion_interval_10+p2p/annotation_images",
    "6_ablation_directinversion_interval_24":"output/ablation_directinversion_interval_24+p2p/annotation_images",
    "6_ablation_directinversion_interval_49":"output/ablation_directinversion_interval_49+p2p/annotation_images",
    # ablation results of different steps
    "7_ablation_directinversion_step_20":"output/ablation_directinversion_step_20+p2p/annotation_images",
    "7_ablation_directinversion_step_100":"output/ablation_directinversion_step_100+p2p/annotation_images",
    "7_ablation_directinversion_step_500":"output/ablation_directinversion_step_500+p2p/annotation_images",
    # ablation results of add target latent
    "8_ablation_directinversion_add-source+p2p":"output/ablation_directinversion_add-source+p2p/annotation_images",
    "8_ablation_directinversion_add-target+p2p":"output/ablation_directinversion_add-target+p2p/annotation_images",
    }


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_mapping_file', type=str, default="/data/data/bim/PnPInversion/PIE-Bench_v1/mapping_file.json") ## ERROR
    parser.add_argument('--metrics',  nargs = '+', type=str, default=[
                                                         "structure_distance",
                                                         "psnr_unedit_part",
                                                         "lpips_unedit_part",
                                                         "mse_unedit_part",
                                                         "ssim_unedit_part",
                                                         "clip_similarity_source_image",
                                                         "clip_similarity_target_image",
                                                         "clip_similarity_target_image_edit_part",
                                                         ])
    parser.add_argument('--src_image_folder', type=str, default="/data/data/bim/PnPInversion/PIE-Bench_v1/annotation_images")
    parser.add_argument('--tgt_methods', nargs = '+', type=str, default=[
                                                                    "1_ddim+p2p", "1_null-text-inversion+p2p_a800",
                                                                    "1_null-text-inversion+p2p_3090", "1_negative-prompt-inversion+p2p",
                                                                    "1_stylediffusion+p2p", "1_directinversion+p2p",
                                                                  ])
    parser.add_argument('--result_path', type=str, default="evaluation_result_refedit.csv")
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--edit_category_list',  nargs = '+', type=str, default=[
                                                                                "0",
                                                                                "1",
                                                                                "2",
                                                                                "3",
                                                                                "4",
                                                                                "5",
                                                                                "6",
                                                                                "7",
                                                                                "8",
                                                                                "9"
                                                                                ]) # the editing category that needed to run
    parser.add_argument('--evaluate_whole_table', action= "store_true") # rerun existing images

    args = parser.parse_args()
    
    annotation_mapping_file=args.annotation_mapping_file
    metrics=args.metrics
    src_image_folder=args.src_image_folder
    tgt_methods=args.tgt_methods
    edit_category_list=args.edit_category_list
    evaluate_whole_table=args.evaluate_whole_table
    
    tgt_image_folders={}
    
    # if evaluate_whole_table:
    #     for key in all_tgt_image_folders:
    #         if key[0] in tgt_methods:
    #             tgt_image_folders[key]=all_tgt_image_folders[key]
    # else:
    #     for key in tgt_methods:
    #         tgt_image_folders[key]=all_tgt_image_folders[key]

    tgt_image_folders = {}
    for i, key in enumerate(tgt_methods):
        key = key.split("/")[-1]
        tgt_image_folders[key]=tgt_methods[i] 
    
    result_path=args.result_path
    
    metrics_calculator=MetricsCalculator(args.device)
    
    with open(result_path,'w',newline="") as f:
        csv_write = csv.writer(f)
        
        csv_head=[]
        for tgt_image_folder_key,_ in tgt_image_folders.items():
            for metric in metrics:
                csv_head.append(f"{tgt_image_folder_key}|{metric}")
        
        data_row = ["file_id", "editing type"]+csv_head
        csv_write.writerow(data_row)

    with open(annotation_mapping_file,"r") as f:
        annotation_file=json.load(f)

    for key, item in annotation_file.items():
        if item["editing_type_id"] not in edit_category_list:
            continue
        print(f"evaluating image {key} ...")
        base_image_path=item["image_path"]
        mask=mask_decode(item["mask"])
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        
        mask=mask[:,:,np.newaxis].repeat([3],axis=2)
        
        src_image_path=os.path.join(src_image_folder, base_image_path)
        src_image = Image.open(src_image_path)
        
        
        evaluation_result=[key, item["editing_type_id"]]
        
        for tgt_image_folder_key,tgt_image_folder in tgt_image_folders.items():
            tgt_image_path=os.path.join(tgt_image_folder, base_image_path)
            print(f"evluating method: {tgt_image_folder_key}")
            
            tgt_image = Image.open(tgt_image_path)
            if tgt_image.size[0] != tgt_image.size[1]:
                # to evaluate editing
                tgt_image = tgt_image.crop((tgt_image.size[0]-512,tgt_image.size[1]-512,tgt_image.size[0],tgt_image.size[1])) 
                # to evaluate reconstruction
                # tgt_image = tgt_image.crop((tgt_image.size[0]-512*2,tgt_image.size[1]-512,tgt_image.size[0]-512,tgt_image.size[1])) 
            
            for metric in metrics:
                print(f"evluating metric: {metric}")
                evaluation_result.append(calculate_metric(metrics_calculator,metric,src_image, tgt_image, mask, mask, original_prompt, editing_prompt))
                        
        with open(result_path,'a+',newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(evaluation_result)
        
        