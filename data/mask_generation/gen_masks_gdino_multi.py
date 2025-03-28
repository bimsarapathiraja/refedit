import os, sys

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

# for path in sys.path:
#     print(path)

import argparse
import copy
import json
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

from huggingface_hub import hf_hub_download

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    args.device = device
    model = build_model(args)
    
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model  

# Load image 
def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(BytesIO(r.content)) as im:
        im.save(image_file_path)
    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))

# detect object using grounding DINO
def detect(image, text_prompt, model, box_threshold = 0.2, text_threshold = 0.25):
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    return boxes, logits, phrases 

def crop_image(image, boxes):
    h, w, _ = image.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    x1, y1, x2, y2 = xyxy[0]
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]
    return cropped_image
  
def segment(image, sam_model, boxes, device='cpu'):
    sam_model.set_image(image)
    H, W, _ = image.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = False,
        )
    return masks.cpu()

def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def return_final_box(detected_boxes, logits, phrases, edit_obj):

    # Filtering the indices where the phrase is 'the right pot'
    filtered_boxes_indices = [i for i, phrase in enumerate(phrases) if edit_obj in phrase]
    # filtered_boxes_indices = [i for i in filtered_boxes_indices if edit_obj in phrases[i]]

    # Extract the boxes and logits based on these indices
    filtered_boxes = [detected_boxes[i] for i in filtered_boxes_indices]
    filtered_logits = [logits[i] for i in filtered_boxes_indices]

    # Now, we select the box with the highest confidence score (logit)
    max_confidence_index = filtered_logits.index(max(filtered_logits))

    # Result: selected box and its confidence score
    selected_box = filtered_boxes[max_confidence_index].unsqueeze(0)
    selected_confidence = filtered_logits[max_confidence_index].unsqueeze(0)

    return selected_box, selected_confidence

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    # ckpt_filenmae = "groundingdino_swint_ogc.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py" 
    # ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py" 

    groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

    sam_checkpoint = 'sam_vit_h_4b8939.pth'

    sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    args = parse_args()
    input_json = args.input_json
    output_dir = args.output_dir

    with open(input_json, "r") as f:
        data = json.load(f)

    # "1": {
    #     "prompt": "In the art gallery, there are three sculptures placed in a row. The leftmost sculpture is a marble bust, the middle one is a bronze abstract piece, and the rightmost is a wooden carving.",
    #     "edit_1": {
    #         "editing_prompt": "In the art gallery, there are three sculptures placed in a row. The leftmost sculpture is a marble bust, the middle one is a gold abstract piece, and the rightmost is a wooden carving.",
    #         "editing_instruction": "Change the material of the middle sculpture to gold",
    #         "editing_object": "Sculpture",
    #         "referring_expression": "The sculpture in the middle",
    #         "descriptive_referring_expression": "The bronze abstract sculpture in the middle"
    #     },
    #     "edit_2": {
    #         "editing_prompt": "In the art gallery, there are three sculptures placed in a row. The leftmost sculpture is a marble bust, the middle one is a bronze abstract piece, and the rightmost is a silver carving.",
    #         "editing_instruction": "Change the material of the rightmost sculpture to silver",
    #         "editing_object": "Sculpture",
    #         "referring_expression": "The sculpture on the rightmost side",
    #         "descriptive_referring_expression": "The wooden carving of the rightmost sculpture"
    #     }
    # },

    # get the keys of the json file and then use that to iterate over the data

    keys = list(data.keys())


    for id in tqdm(keys):
        # id = i + 1
        print(id)
        info = data[id]
        img_dir_pth = f"{output_dir}/{id}"
        img_names = os.listdir(img_dir_pth)
        img_name = [x for x in img_names if x.startswith("initial_")][0]
        initial_img_pth = f"{img_dir_pth}/{img_name}"
        image_source, image = load_image(initial_img_pth)

        try:
            edit_objs = [info["edit_1"]['editing_object'], info["edit_2"]['editing_object'], info["edit_3"]['editing_object'], info["edit_4"]['editing_object'], info["edit_5"]['editing_object']]
            descriptive_expressions = [info["edit_1"]['descriptive_referring_expression_single_object'], info["edit_2"]['descriptive_referring_expression_single_object'], info["edit_3"]['descriptive_referring_expression_single_object'], info["edit_4"]['descriptive_referring_expression_single_object'], info["edit_5"]['descriptive_referring_expression_single_object']]
        except:
            print("No edit instructions provided:", img_dir_pth)
            continue

        img_save_dir_pth = img_dir_pth
        # final_boxes = []

        for edit_obj, ref_exp in zip(edit_objs, descriptive_expressions):
            edit_obj = edit_obj.lower()
            ref_exp = ref_exp.lower()

            img_save_dir_sub_pth = f"{img_save_dir_pth}/{ref_exp.replace(' ', '_')}"
            os.makedirs(img_save_dir_sub_pth, exist_ok=True)

            detected_boxes, logits, phrases = detect(image, text_prompt=ref_exp, model=groundingdino_model)

            # sort the boxes based on the confidence score
            sorted_indices = torch.argsort(logits, descending=True)
            sorted_boxes = detected_boxes[sorted_indices]

            for i, box in enumerate(sorted_boxes):
                box_id_pth = f"{img_save_dir_sub_pth}/{i+1}"
                os.makedirs(box_id_pth, exist_ok=True)

                box = box.unsqueeze(0)

                annotated_frame = annotate(image_source=image_source, boxes=box, logits=logits[i].unsqueeze(0), phrases=[phrases[i]])
                annotated_frame = annotated_frame[...,::-1]
                Image.fromarray(annotated_frame).save(f"{box_id_pth}/bbox.png")

                # crop and save the image
                cropped_image = crop_image(image_source, box)
                Image.fromarray(cropped_image).save(f"{box_id_pth}/cropped.png")

                segmented_frame_masks = segment(image_source, sam_predictor, boxes=box, device=device)
                annotated_frame_with_mask = draw_mask(segmented_frame_masks[0][0], annotated_frame)
                Image.fromarray(annotated_frame_with_mask).save(f"{box_id_pth}/seg.png")

                mask = segmented_frame_masks[0][0].cpu().numpy()
                Image.fromarray(mask).save(f"{box_id_pth}/mask.png")

if __name__ == "__main__":
    main()

# python gen_masks_gdino_multi.py --input_json change_color.json --output_dir change_color