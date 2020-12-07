import os
import glob
import numpy as np
import random
from PIL import Image

import torch
import torchvision
from torchvision import transforms

class Dataset(object):
    '''
        Generate dataset from given dataset/label file
        Return a map-style dataset which can be used with torch.utils.data.DataLoader
    '''
    def __init__(self, dataset_path, class_name_path=None, transform=None):
        '''
            Args:
                dataset_path: dataset file path (can be created by list_all_image function)
                    * dataset file format:
                        absolute_path/to/image_0 0 (means label 0)
                        absolute_path/to/image_1 0 (means label 0)
                        absolute_path/to/image_2 1 (means label 1)
                class_name_path: class name file path
                    * class name file format:
                        class_name_0
                        class_name_1
                        class_name_2
                transform: transform function, if not None it will be applied to the output images on loading
        '''

        self.transform = transform

        # load dataset
        assert os.path.isfile(dataset_path), f"{dataset_path} not exist!"
        with open(dataset_path, "r") as f:
            content = f.readlines()
        assert len(content[0].split()) == 2, "The dataset file input is invalid!"
        self.data = [i.strip().split() for i in content]

        # load class names
        class_nums = len(set([i[1] for i in self.data]))
        if class_name_path is not None:
            with open(class_name_path, "r") as f:
                content = f.readlines()
            class_names = [i.strip() for i in content]
            assert len(class_names) == class_nums, "the num of classes in the index_name_map file is not equal to the num of classes in the image list file of the dataset!"
        else:
            class_names = ["class "+str(i) for i in range(class_nums)]
        self.classes = class_names

    def __getitem__(self, i):
        img_path, label = self.data[i]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, int(label)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Dataset({len(self.classes)} classes and {len(self.data)} images)"

def list_all_image(dataset_dir, output_path, class_list=None):
    '''
        List all images in dataset directory to one txt file. The first column is the relative path of each image and the second column is its label.

        Args:
            dataset_dir: path to dataset directory
                valid directory structure:
                |-- dataset
                    |-- class 1
                        |-- img1.jpg
                        |-- img2.jpg  
                    |-- class 2
                        |-- img1.jpg
                        |-- img2.jpg
            output_path: the path to save the output file (eg: "./train_dataset.txt")
            class_list: class to list, if none list all classes of images, if [1,2] only list images belonging to class 1 and class 2
    '''
    
    # get class dir  
    classes = glob.glob(os.path.join(dataset_dir, "*"))
    
    # list all image
    output_list = []
    label_list = []
    checked_label_count = 0
    for class_ in classes:
        if os.path.isdir(class_):
            if (class_list is None) or (checked_label_count in class_list):
                label_name = os.path.basename(class_)
                label_index = len(label_list)
                label_list.append(label_name)
                
                image_count = 0
                for img_path in glob.glob(os.path.join(class_,"*.jpg")):
                    image_count += 1
                    output_list.append((img_path, str(label_index)))
                for img_path in glob.glob(os.path.join(class_,"*.png")):
                    image_count += 1
                    output_list.append((img_path, str(label_index)))

                print(f"\r{len(label_list)}/{len(classes) if class_list is None else len(class_list)}   ", end="")
            else:
                pass
            checked_label_count += 1

    print(f"\rtotal {len(label_list)} classes {len(output_list)} imgs")

    # write output list
    with open(output_path, "w") as f:
        f.write("\n".join([" ".join(output_list[i]) for i in range(len(output_list))]))
    class_name_path = os.path.join(os.path.dirname(output_path), "class_name.txt")
    with open(class_name_path, "w") as f:
        f.write("\n".join(label_list))
    
    return output_path, class_name_path
