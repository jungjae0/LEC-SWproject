import cv2
import os
import numpy as np
from pathlib import Path
import json
import shutil
import random

def create_yolo_segmentation_format(label_path, output_dir, index, class_id=0):
    """
    Convert label images (each plant instance is segmented) to YOLOv8 segmentation format.

    Parameters:
    - label_path: Path to the directory containing label images.
    - output_dir: Directory to save the YOLO format files.
    - index: Index for folder naming to separate each dataset.
    - class_id: Class ID to assign (e.g., 0 for a single class).
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    label_output_dir = output_dir / f"labels/{index}"
    label_output_dir.mkdir(parents=True, exist_ok=True)

    # Process each label image
    for label_file in Path(label_path).glob("*_label.png"):
        # Read label image in grayscale
        label_img = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
        if label_img is None:
            print(f"Could not read {label_file}, skipping.")
            continue

        # Find contours (objects in the mask)
        contours, _ = cv2.findContours(label_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare the output file for this image
        yolo_label_file = label_output_dir / f"{label_file.stem}.txt"

        with open(yolo_label_file, "w") as f:
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                x_center, y_center = (x + w / 2) / label_img.shape[1], (y + h / 2) / label_img.shape[0]
                width, height = w / label_img.shape[1], h / label_img.shape[0]

                # Get segmentation points (relative to image size)
                segmentation_points = [
                    f"{pt[0][0] / label_img.shape[1]:.6f} {pt[0][1] / label_img.shape[0]:.6f}"
                    for pt in contour
                ]
                segmentation_str = " ".join(segmentation_points)

                # Write to YOLO format: class_id x_center y_center width height segmentation_points
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {segmentation_str}\n")

        print(f"Processed {label_file.name} -> {yolo_label_file}")

# def create_yolo_segmentation_format(label_path, output_dir, index,class_id=0):
#     """
#     Convert label images (each plant instance is segmented) to YOLOv8 segmentation format.
#
#     Parameters:
#     - label_path: Path to the directory containing label images.
#     - output_dir: Directory to save the YOLO format files.
#     - class_id: Class ID to assign (e.g., 0 for a single class).
#     """
#     # Ensure output directories exist
#     output_dir = Path(output_dir)
#     (output_dir / "labels/train").mkdir(parents=True, exist_ok=True)
#
#     # Process each label image
#     for label_file in Path(label_path).glob("*_label.png"):
#         # Read label image
#         label_img = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
#
#         # Find contours (objects in the mask)
#         contours, _ = cv2.findContours(label_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         # Prepare the output file for this image
#         yolo_label_dir = output_dir / f"labels/{index}"
#         yolo_label_dir.mkdir(exist_ok=True)
#         yolo_label_file = os.path.join(yolo_label_dir, f"{label_file.stem}.txt")
#
#         with open(yolo_label_file, "w") as f:
#             for contour in contours:
#                 # Get bounding box
#                 x, y, w, h = cv2.boundingRect(contour)
#                 x_center, y_center = (x + w / 2) / label_img.shape[1], (y + h / 2) / label_img.shape[0]
#                 width, height = w / label_img.shape[1], h / label_img.shape[0]
#
#                 # Get segmentation points (relative to image size)
#                 segmentation_points = [
#                     f"{pt[0][0] / label_img.shape[1]:.6f},{pt[0][1] / label_img.shape[0]:.6f}"
#                     for pt in contour
#                 ]
#                 segmentation_str = " ".join(segmentation_points)
#
#                 # Write to YOLO format: class_id x_center y_center width height segmentation_points
#                 f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {segmentation_str}\n")
#
#         print(f"Processed {label_file.name} -> {yolo_label_file}")


def copy_rgb_images(src_base_dir, dest_base_dir, index):

    src_dir = Path(src_base_dir) / str(index)
    dest_dir = Path(dest_base_dir) / str(index)
    dest_dir.mkdir(parents=True, exist_ok=True)

    for file_path in src_dir.glob('*_rgb.png'):
        dest_file_path = dest_dir / file_path.name
        shutil.copy(file_path, dest_file_path)
        print(f"Copied '{file_path}' to '{dest_file_path}'")


def split_train_val(images_dir, labels_dir, train_ratio=0.8):
    """
    Split images and labels into train and valid sets.

    Parameters:
    - images_dir: Path to the images folder (e.g., 'images/{index}').
    - labels_dir: Path to the labels folder (e.g., 'labels/{index}').
    - train_ratio: Ratio of training set (e.g., 0.8 for 80% training and 20% valid).
    """
    # Define train and valid directories
    train_images_dir = Path(images_dir).parent / "train" / Path(images_dir).name
    val_images_dir = Path(images_dir).parent / "val" / Path(images_dir).name
    train_labels_dir = Path(labels_dir).parent / "train" / Path(labels_dir).name
    val_labels_dir = Path(labels_dir).parent / "val" / Path(labels_dir).name

    # Create directories if they don't exist
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # List all image files
    image_files = list(Path(images_dir).glob("*.png"))

    # Shuffle and split the files
    random.shuffle(image_files)
    split_index = int(len(image_files) * train_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    # Copy files to train and val folders
    for file_path in train_files:
        shutil.copy(file_path, train_images_dir / file_path.name)
        label_file = Path(labels_dir) / file_path.with_suffix(".txt").name
        if label_file.exists():
            shutil.copy(label_file, train_labels_dir / label_file.name)

    for file_path in val_files:
        shutil.copy(file_path, val_images_dir / file_path.name)
        label_file = Path(labels_dir) / file_path.with_suffix(".txt").name
        if label_file.exists():
            shutil.copy(label_file, val_labels_dir / label_file.name)

    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")


def split_data(index):
    # 1. images/{index} 폴더에 있는 file 이름 가져오기
    images_dir = f"../send_code/02_yolo_segmentation_model/yolo_dataset/images/{index}"
    labels_dir = f"../send_code/02_yolo_segmentation_model/yolo_dataset/labels/{index}"
    image_files = [f.split("_")[0] for f in os.listdir(images_dir)]

    random.shuffle(image_files)

    split_index = int(len(image_files) * 0.8)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]


    train_img_dir = "../send_code/02_yolo_segmentation_model/yolo_dataset/train/images"
    val_img_dir = "../send_code/02_yolo_segmentation_model/yolo_dataset/valid/images"
    train_label_dir = "../send_code/02_yolo_segmentation_model/yolo_dataset/train/labels"
    val_label_dir = "../send_code/02_yolo_segmentation_model/yolo_dataset/valid/labels"
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)

    def copy_file(new_img_dir, new_label_dir, files):
        for f in files:
            img_file = f'{f}_rgb.png'
            label_file = f'{f}_label.txt'
            
            origin_img_path = os.path.join(images_dir, img_file)
            origin_label_path = os.path.join(labels_dir, label_file)
            
            new_img_path = os.path.join(new_img_dir, f"{index}_{f}.png")
            new_label_path = os.path.join(new_label_dir, f"{index}_{f}.txt")

            shutil.copy(origin_img_path, new_img_path)
            shutil.copy(origin_label_path, new_label_path)
    copy_file(train_img_dir, train_label_dir, train_files)
    copy_file(val_img_dir, val_label_dir, val_files)

    # shutil.copy(os.path.join(src_folder, file), os.path.join(dest_folder, file))






    # if images_files != labels_files:
    #     print(f"{index}의 label과 image 일치하지 않음")
    # else:
    #     print(f"{index}의 label과 image 일치")



def main():

    # Example usage
    indexs = ['A1', 'A2', 'A3', 'A4']
    for index in indexs:
        label_dir = rf"./training/{index}"
        #
        output_directory = r"yolo_dataset\images"
        # create_yolo_segmentation_format(label_dir, output_directory, index)
        # copy_rgb_images(label_dir, output_directory, index)
        split_data(index)






if __name__ == '__main__':
    main()