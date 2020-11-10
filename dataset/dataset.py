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
        Generate dataset from given dataset txt file and label file
    '''
    def __init__(self, dataset_path, index_name_map_path=None, transform=None):
        '''
            Args:
                dataset_path: dataset txt file path (can be created by list_dataset function)
                    * dataset file format:
                        absolute_path/to/image_0 0 (means label 0)
                        absolute_path/to/image_1 0 (means label 0)
                        absolute_path/to/image_2 1 (means label 1)
                index_name_map_path: index_name_map txt file path
                    * label file format:
                        label_name_0
                        label_name_1
                        label_name_2
                transform: transform on images
        '''

        self.transform = transform

        with open(dataset_path, "r") as f:
            content = f.readlines()

        # for debug
        # print("[+] load dataset from path {}".format(dataset_path))

        # if in content there is only the image path without any label, add default label 0 to it
        self.data = [i.strip().split() if len(i.strip().split()) == 2 else [i.strip(), 0] for i in content]
        class_nums = len(set([i[1] for i in self.data]))
        # print(self.data[0:10])

        # for debug
        # print("[+] total {} images".format(len(self.data)))
        # print(" - {} is the first image, its label is {}".format(self.data[0][0], self.data[0][1]))
        # print(" - {} is the second image, its label is {}".format(self.data[1][0],self.data[1][1]))
        # print(" - {} is the third image, its label is {}".format(self.data[2][0], self.data[2][1]))
        
        if index_name_map_path is not None:
            with open(index_name_map_path, "r") as f:
                content = f.readlines()
            class_names = [i.strip() for i in content]
            assert len(class_names) == class_nums, "the num of classes in the index_name_map file is not equal to the num of classes in the image list file of the dataset!"
        else:
            class_names = ["class "+str(i) for i in range(class_nums)]

        # for debug
        # print("[+] load label names")
        # if len(content) < 200:
        #     for (idx,name) in enumerate(class_names):
        #         print(" - label {}: {}".format(idx, name))
        # else:
        #     for (idx,name) in enumerate(class_names[:200]):
        #         print(" - label {}: {}".format(idx, name))
        self.classes = class_names

        # print("[+] create dataset")

    def __getitem__(self, i):
        img_path, label = self.data[i]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        # img.show()
        return img, int(label)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Dataset({len(self.classes)} classes and {len(self.data)} images)"


# class PathDataset(Dataset):
#     '''
#         PathDataset output image,path instead of image,label
#     '''

def list_dataset(dataset_dir, output_path, class_select=None):
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
            output_path: the path of the generated txt file (including its name)
            class_select: select class index, if none select all classes, if [1,2] only list images of class 1 and class 2
    '''
    
    output_list = [] # the final output list
    
    label_list = [] # all label names
    checked_label_count = 0
    # label_count = 0
    
    classes = glob.glob(os.path.join(dataset_dir, "*"))
    
    for class_ in classes:
        if os.path.isdir(class_):
            if (class_select is None) or (checked_label_count in class_select):
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

                print(f"\r{len(label_list)}/{len(classes) if class_select is None else len(class_select)}   ", end="")
            else:
                pass
            checked_label_count += 1

    print(f"\rtotal {len(label_list)} classes {len(output_list)} imgs")

    # write output list
    with open(output_path, "w") as f:
        f.write("\n".join([" ".join(output_list[i]) for i in range(len(output_list))]))
    # write the index_to_name_map of label
    with open(output_path.replace(".txt","_index_to_name_map.txt"), "w") as f:
        f.write("\n".join(label_list))
    index_name_map_path = output_path.replace(".txt","_index_to_name_map.txt")
    
    return output_path, index_name_map_path

def generate_dataset(dataset_dir, create_test=True, transform_train=None, transform_test=None, class_select=None):
    print("=> list dataset")
    train_path, test_path, label_name_path = list_dataset(dataset_dir, "./dataset.txt", create_test=create_test, class_select=class_select)
    print()
    print("=> generate train dataset")
    dataset_train = Dataset(train_path, label_name_path, transform_train)
    if create_test:
        print()
        print("=> generate test dataset")
        dataset_test = Dataset(test_path, label_name_path, transform_test)
        return dataset_train, dataset_test
    else:
        return dataset_train
