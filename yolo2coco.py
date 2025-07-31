import os
import cv2
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='D:\\code\\python\\yolov5-prune-light\\yolov8-prune\\datasets\\AI-TOD', type=str,
                    help="root path of images and labels, include ./images and ./labels and classes.txt")
parser.add_argument('--save_path', type=str, default='D:\\code\\python\\yolov8-prune-light\\yolov5-prune\\datasets\\AI-TOD\\instances_test2017.json',
                    help="if not split the dataset, give a path to a json file")

arg = parser.parse_args()

def yolo2coco(arg):
    root_path = arg.root_dir
    print("Loading data from ", root_path)

    assert os.path.exists(root_path)
    originLabelsDir = os.path.join(root_path, 'labels/test2017')
    originImagesDir = os.path.join(root_path, 'images/test2017')
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = list(map(lambda x: x.strip(), f.readlines()))
    # images dir name
    indexes = os.listdir(originImagesDir)

    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        txtFile = index.replace('images', 'txt').replace('.jpg', '.txt').replace('.png', '.txt')
        im = cv2.imread(os.path.join(originImagesDir, index))
        height, width, _ = im.shape
        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            continue
        dataset['images'].append({'file_name': index,
                                  'id': int(index[:-4]) if index[:-4].isnumeric() else index[:-4],
                                  'width': width,
                                  'height': height})
        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W, _ = im.shape
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                cls_id = int(label[0])
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': int(index[:-4]) if index[:-4].isnumeric() else index[:-4],
                    'iscrowd': 0,
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    with open(arg.save_path, 'w') as f:
        json.dump(dataset, f)
        print('Save annotation to {}'.format(arg.save_path))


if __name__ == "__main__":
    yolo2coco(arg)
