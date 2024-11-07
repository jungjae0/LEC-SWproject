import cv2
import torch
import numpy as np
from PIL import Image

from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple


from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from ultralytics import YOLO
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


# ----------- Plant Segmentation With CLIPSeg Model
def with_clip(image):
    # 전체 이미지에서 'plant'를 detection & segmentation

    # 필요한 모델 정의(CLIPSeg)
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    segementator = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

    prompts = ["a plant."]

    # text-img 쌍의 모델 입력값 정의
    inputs = processor(text=prompts, images=[image] * len(prompts), padding="max_length", return_tensors="pt")

    # 예측
    with torch.no_grad():
        outputs = segementator(**inputs)

    preds = outputs.logits.unsqueeze(1)

    heat_map = torch.sigmoid(preds[0][0]).detach().cpu().numpy()
    heat_map_resized = cv2.resize(heat_map, (image.size[0], image.size[1]))
    mask = (heat_map_resized > 0.5).astype(np.uint8)
    masked_img = cv2.bitwise_and(np.array(image), np.array(image), mask=mask)

    return masked_img


# ----------- Plant Segmentation With YOLO Model
def with_yolo(image):
    yolo_plant_seg_model = YOLO(r'./weights/segmentation_yolo_model.pt')  # 학습된 모델 경로
    results = yolo_plant_seg_model.predict(source=image, conf=0.25, save=False, save_txt=False)

    r = results[0]

    mask = r.masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = (mask * 255).astype(np.uint8)

    segmented = cv2.bitwise_and(image, image, mask=mask)

    return segmented

# ----------- Plant Segmentation With Grounded-SAM Model
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

def get_boxes(results: DetectionResult) -> List[List[List[float]]]:
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)

    return [boxes]


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea)

    polygon = largest_contour.reshape(-1, 2).tolist()

    return polygon

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)

    pts = np.array(polygon, dtype=np.int32)

    cv2.fillPoly(mask, [pts], color=(255,))

    return mask

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False) -> List[np.ndarray]:
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks



def detect(
    image: Image.Image,
    labels: List[str],
    threshold: float = 0.3,
    detector_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label+"." for label in labels]

    results = object_detector(image,  candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results

def segment(image, detection_results, polygon_refinement, segmenter_id):
    """
    Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results


def grounded_segmentation(
    image: Union[Image.Image, str],
    labels: List[str],
    threshold: float = 0.3,
    polygon_refinement: bool = False,
    detector_id: Optional[str] = None,
    segmenter_id: Optional[str] = None
):
    # if isinstance(image, str):
    #     image = load_image(image)

    detections = detect(image, labels, threshold, detector_id)
    detections = segment(image, detections, polygon_refinement, segmenter_id)

    return np.array(image), detections

def annotate(image, detection_results):
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    detection = detection_results[0]
    mask = detection.mask

    segmented = cv2.bitwise_and(image_cv2, image_cv2, mask=mask)

    return segmented

def with_groundedsam(image):
    labels = ["a leaf."]
    threshold = 0.3
    detector_id = "IDEA-Research/grounding-dino-tiny"
    segmenter_id = "facebook/sam-vit-base"

    image_array, detections = grounded_segmentation(
        image=image,
        labels=labels,
        threshold=threshold,
        polygon_refinement=True,
        detector_id=detector_id,
        segmenter_id=segmenter_id
    )

    plant_object = annotate(image_array, detections)

    return plant_object