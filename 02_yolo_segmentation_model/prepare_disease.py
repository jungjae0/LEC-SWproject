import os
import cv2
import glob
import tqdm

def create_yolo_segmentation_format(label_file, output_dir, class_id=0):

    label_img = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
    if label_img is None:
        print(f"Could not read {label_file}, skipping.")

    # Find contours (objects in the mask)
    contours, _ = cv2.findContours(label_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare the output file for this image
    yolo_label_file = os.path.join(output_dir, label_file.split("\\")[-1].replace(".png", ".txt"))

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

    print(f"Processed {label_file} -> {yolo_label_file}")

def main():
    data_dir = "../03_yolo_disease/disease_dataset"

    mask_dirs = glob.glob(f"{data_dir}/*_masks")
    for mask_dir in mask_dirs:
        index = mask_dir.split("\\")[-1].split("_")[0]
        output_dir = os.path.join(data_dir, index, "labels")
        os.makedirs(output_dir, exist_ok=True)
        mask_files = glob.glob(f"{mask_dir}/*.png")
        for mask_file in tqdm.tqdm(mask_files, total=len(mask_files), desc=f"{index}"):
            create_yolo_segmentation_format(mask_file, output_dir)





if __name__ == '__main__':
    main()