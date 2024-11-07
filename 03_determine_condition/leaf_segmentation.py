import numpy as np

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def with_sam(plant_object):
    sam_checkpoint = "./weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # img를 모델의 input으로 주어 segmentation의 수행
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.95,
        min_mask_region_area=900
    )

    masks = mask_generator.generate(plant_object)
    seg_masks = [mask['segmentation'].astype(np.uint8) for mask in masks]

    return seg_masks