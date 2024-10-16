import os

# Define paths for YOLO dataset annotations
yolo_annotations_dir = r'dataset\test\class_labels'
output_txt = r'dataset\test\labels'

for txt_file in os.listdir(yolo_annotations_dir):
# Load YOLO annotations (format: image_path x_center y_center width height class_id)
    with open(os.path.join(yolo_annotations_dir, txt_file), 'r') as f:
        yolo_annotations = f.readlines()

    # Open output file for segmentation labels
    with open(os.path.join(output_txt, txt_file), 'w') as output_file:
        for annotation in yolo_annotations:
            annotation = annotation.split()
            # image_path = annotation[0]
            class_id = annotation[0]

            # Extract bounding box coordinates
            x_center, y_center, width, height = map(float, annotation[1:5])
            x_min, y_min = (x_center - width / 2), (y_center - height / 2)
            x_max, y_max = (x_center + width / 2), (y_center + height / 2)

            # Convert bounding box coordinates to polygon format
            poly_coords = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max)
            ]

            # Write polygon coordinates and class label to output file
            # output_file.write(f'{image_path},')
            output_file.write(f'{class_id} ')
            for point in poly_coords:
                output_file.write(f'{point[0]} {point[1]} ')
            output_file.write(f'\n')
