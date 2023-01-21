import subprocess
import cv2
import numpy as np
import os
import csv
import pandas as pd
import shutil


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def create_yolo_label_files(images_folder_path, csv_file_path):
    all_images_names = os.listdir(images_folder_path)
    # print(all_images_names)
    df = pd.read_csv(csv_file_path)
    print(df.columns)

    for image_name in all_images_names:
        image_path = "images/" + image_name
        image = cv2.imread(image_path)
        im_height, im_width = image.shape[:2]

        all_objects = df.loc[df['image_path'] == image_name]
        yolo_objects = []
        # print(all_objects.to_numpy())
        all_objects = all_objects.to_numpy()
        obj_count = all_objects.shape[0]
        label_path = "labels/" + image_name.split('.')[0] + ".txt"
        label_file = open(label_path, "w")
        yolo_content = []
        for i in range(obj_count):
            [xmax, xmin, ymax, ymin] = all_objects[i, 3:7]
            bbox = [int(xmin * 2), int(ymin * 2), int(xmax * 2), int(ymax * 2)]

            yolo_bbox = xml_to_yolo_bbox(bbox, im_width, im_height)
            yolo_bbox.insert(0, int(all_objects[i, 0]))

            yolo_bbox = [str(x) + " " for x in yolo_bbox]
            yolo_bbox.append("\n")
            label_file.writelines(yolo_bbox)

        label_file.close()


def split_train_test(images_folder_path, labels_folder_path):
    all_images_names = os.listdir(images_folder_path)
    count_of_images = len(all_images_names)

    df = pd.read_csv("test.csv")

    for image_name in all_images_names:
        if image_name in df['image_path'].values:
            shutil.move(images_folder_path + "/" + image_name, "test/images/" + image_name)
            label_name = image_name.split('.')[0] + ".txt"
            shutil.move(labels_folder_path + "/" + label_name, "test/labels/" + label_name)
        else:
            shutil.move(images_folder_path + "/" + image_name, "train/images/" + image_name)
            label_name = image_name.split('.')[0] + ".txt"
            shutil.move(labels_folder_path + "/" + label_name, "train/labels/" + label_name)


def split_train_valid(train_folder_path, valid_folder_path, split_percentage=0.2):
    all_image_names = os.listdir(train_folder_path + "/images")
    total_count_of_images = len(all_image_names)
    valid_count = int(total_count_of_images * split_percentage)
    image_count = 0
    for image_name in all_image_names:
        if image_count <= valid_count:
            shutil.move(train_folder_path + "/images/" + image_name, valid_folder_path + "/images/" + image_name)
            label_name = image_name.split('.')[0] + ".txt"
            shutil.move(train_folder_path + "/labels/" + label_name, valid_folder_path + "/labels/" + label_name)
            image_count += 1
        else:
            break


def get_class_names(csv_file_path):
    df = pd.read_csv(csv_file_path)
    total_classes = len(df['class'].unique())
    class_names = []
    for i in range(total_classes):
        selection = df.loc[df['class'] == i]
        # print(selection["name"].unique())
        class_names.append(selection["name"].unique()[0])

    print(class_names)
    return class_names


# split_train_test("images", "labels")
# split_train_valid("train", "valid")
get_class_names("train.csv")
