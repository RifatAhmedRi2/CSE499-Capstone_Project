import os
import json
import cv2 
import shutil
from shapely.geometry import Polygon
from shapely.wkt import loads

LABEL_DIR = r'dataset\train\jsons'
SAVE_DIR = r'dataset\train\labels'
IMG_DIR = r'dataset\train\images'
BASE_DIR = r'dataset\train\all_images'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

damage_dict = {
    'no-damage' : 0,
    'minor-damage' : 1,
    'major-damage' : 2,
    'destroyed' : 3,
    'un-classified' : 4
}

for json_name in os.listdir(LABEL_DIR):
    if 'post_disaster' in json_name:
        print(f'Processing Json -> {json_name}')
        with open(os.path.join(LABEL_DIR, json_name), 'r') as f:
            data = json.load(f)
        metadata = data['metadata']
        xys = data['features']['xy']
        if len(xys) != 0:
            img = cv2.imread(os.path.join(r'dataset\train\all_images', json_name[:-5] + '.png'))
            shutil.copyfile(os.path.join(BASE_DIR, json_name[:-5] + '.png'), os.path.join(IMG_DIR, json_name[:-5] + '.png'))
            lines = []
            for xy in xys:
                coordinates = list(loads(xy['wkt']).exterior.coords)
                max_x, min_x = round(max(coordinates, key=lambda x: x[0])[0]), round(min(coordinates, key=lambda x: x[0])[0])
                max_y, min_y = round(max(coordinates, key=lambda x: x[1])[1]), round(min(coordinates, key=lambda x: x[1])[1])
                img = cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255,0,0), 3)
                line = str(damage_dict[xy['properties']['subtype']])
                x, y = round((max_x + min_x) / 2), round((max_y + min_y)/2)
                width, height = max_x - min_x, max_y - min_y

                line += f" {str(x/metadata['original_width'])} {str(y/metadata['original_height'])} {str(width/metadata['original_width'])} {str(height/metadata['original_height'])}"
                lines.append(line)
            
            with open(os.path.join(SAVE_DIR, json_name[:-5] + '.txt'), 'w') as file:
                for line in lines:
                    file.write(line + '\n')

        # cv2.imwrite(json_name[:-5] + '.png', img)
        # break