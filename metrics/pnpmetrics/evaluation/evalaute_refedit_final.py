import json
import argparse
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import csv
from matrics_calculator import MetricsCalculator
from torchvision import transforms
import random

# def mask_decode(encoded_mask,image_shape=[512,512]):
#     length=image_shape[0]*image_shape[1]
#     mask_array=np.zeros((length,))
    
#     for i in range(0,len(encoded_mask),2):
#         splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
#         for j in range(splice_len):
#             mask_array[encoded_mask[i]+j]=1
            
#     mask_array=mask_array.reshape(image_shape[0], image_shape[1])
#     # to avoid annotation errors in boundary
#     mask_array[0,:]=1
#     mask_array[-1,:]=1
#     mask_array[:,0]=1
#     mask_array[:,-1]=1
            
#     return mask_array

def mask_decode(encoded_mask, base_image_path):
    
    org_image_path = base_image_path.replace("_512", "")
    org_image_path = f"DATA_PATH/{org_image_path}"

    org_image = Image.open(org_image_path)
    org_image_size = org_image.size
    
    # Create a blank image
    new_mask = Image.new('L', org_image_size, 0)
    new_draw = ImageDraw.Draw(new_mask)

    # Draw the polygon
    new_draw.polygon(encoded_mask, outline=1, fill=1)

    # Convert to numpy array for further processing if needed
    new_mask_array = np.array(new_mask)
    
    new_mask_array = cv2.resize(new_mask_array, (512, 512))
    
    new_mask_array = cv2.dilate(new_mask_array, np.ones((15, 15), np.uint8), iterations=1)
    
    new_mask_array[0,:]=1
    new_mask_array[-1,:]=1
    new_mask_array[:,0]=1
    new_mask_array[:,-1]=1
    
    # save the mask
    # random_num = random.randint(0, 100)
    # cv2.imwrite("/home/cr8dl-user/DATA_bim/kingkong2/PnPInversion/run_eval/mask_" + str(random_num) + ".jpg", new_mask_array*255)
    
    return new_mask_array

def calculate_metric(metrics_calculator,metric, src_image, tgt_image, src_mask, tgt_mask,src_prompt,tgt_prompt):
    try:
        
        src_image = transforms.Resize(tgt_image.size[::-1])(src_image)
        src_mask = np.array(Image.fromarray(src_mask.astype(np.uint8)).resize(tgt_mask.shape[:2][::-1], Image.NEAREST))
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
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_mapping_file', type=str, default="/home/cr8dl-user/DATA_bim/kingkong2/refcoco/final_benchmark/combined_data_512.json")
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
    parser.add_argument('--src_image_folder', type=str, default="/home/cr8dl-user/DATA_bim/kingkong2/refcoco/final_benchmark")
    parser.add_argument('--tgt_methods', nargs = '+', type=str, default=[
                                                                    "1_ddim+p2p", "1_null-text-inversion+p2p_a800",
                                                                    "1_null-text-inversion+p2p_3090", "1_negative-prompt-inversion+p2p",
                                                                    "1_stylediffusion+p2p", "1_directinversion+p2p",
                                                                  ])
    parser.add_argument('--result_path', type=str, default="evaluation_result_refedit_new.csv")
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
        # if item["editing_type_id"] not in edit_category_list:
        #     continue
        editing_type = item["image_path"].split("/")[0]
        print(f"evaluating image {key} ...")
        base_image_path=item["image_path"]
        image_name = item["image_path"].split("/")[-1].split(".")[0]
        image_name = f"{image_name}_{key}.jpg"
        base_image_tgt_path = os.path.join('/'.join(item["image_path"].split("/")[:-1]), image_name)
        # mask = [int(x) for x in item["mask"][0]]
        # mask=mask_decode(mask)
        mask=mask_decode(item["mask"], base_image_path)
        # original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        
        mask=mask[:,:,np.newaxis].repeat([3],axis=2)
        
        src_image_path=os.path.join(src_image_folder, base_image_path)
        src_image = Image.open(src_image_path)
        
        
        evaluation_result=[key, editing_type]
        
        for tgt_image_folder_key,tgt_image_folder in tgt_image_folders.items():
            tgt_image_path=os.path.join(tgt_image_folder, base_image_tgt_path)
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