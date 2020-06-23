import cv2
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET

def get_image_paths():
    folder = './images'
    files = os.listdir(folder)
    files.sort()
    files = ['{}/{}'.format(folder, file) for file in files]
    return files

def expand_dataset(path_to_folder):
    """
    Expands the image dataset with scaling, addition of noise, and label modifications
    """
    path_to_augmented_folder = os.path.join(path_to_folder, 'augmented_set')
    
    #create new augmented_set folder in the image folder
    if not os.path.isdir(path_to_augmented_folder):
        os.mkdir(path_to_augmented_folder)

    extension = "*.jpg" or "*.png" or "*.JPG" or "*.JPEG" or "*.jpeg"

    for file in glob.glob(os.path.join(path_to_folder, extension)):
        print(f"modifying image: {file}")
        path_to_image = file
        name_without_extension = path_to_image.split('/')[-1].split('.')[0]
        path_to_xml = os.path.join(path_to_folder, name_without_extension) + '.xml'
        print(f"modifying xml: {path_to_xml}")
        #This section will iterate through the scaling factors and then create an image/xml scaled to that size
        for repeat in [1, 2]:
            if repeat == 1:
                resized_image = np.flipud(path_to_image.img)
            elif repeat == 2:
                resized_image = np.fliplr(path_to_image.img)
            elif repeat == 3:
                resized_image = np.fliplr(np.flipud(path_to_image.img))
            # newheight, newwidth = resized_image.shape[:2]
            cv2.imwrite(os.path.join(path_to_augmented_folder, name_without_extension+"_flipped_"+str(repeat)+'.jpg'), resized_image)
            #copies xml from original folder to augmented folder with resized and name of scaling done
            copyfile(path_to_xml, os.path.join(path_to_augmented_folder, name_without_extension)+"_flipped_"+str(repeat)+'.xml')
            path_to_new_xml = os.path.join(path_to_augmented_folder, name_without_extension)+"_flipped_"+str(repeat)+'.xml'
            if os.path.exists(path_to_new_xml):
                if self.kwargs["how"] == "vertical":
                    for box in self.annotations["boxes"]:
                    name = box['label']
                    # unchanged
                    width = box["width"]
                    # unchanged
                    height = box["height"]
                    # check if annotation is on bottom of image
                    if box["y"] < img_height/2:
                        # new Y position is addition of center and diff of
                        # (center and old y)
                        nY = img_height/2 + (img_height/2 - box["y"])
                    # else annotation is on right side of image
                    elif box["y"] >= img_height/2:
                        # new Y position is center  diff between old x and center
                        nY = img_height/2 - (box["y"] - img_height/2)
                    # unchanged
                    nX = box["x"]
                    # create new boxes for each label
                    new_box.append({
                    "label": name,
                    "x": nX,
                    "y": nY,
                    "width": width,
                    "height": height
                })
                change_xml_filename(path_to_new_xml, name_without_extension+"_flipped_"+str(repeat)+'.jpg')
            else:
                print("path to new xml does not exist")


def change_xml_filename(xml_path, newfilename):
    if type(xml_path) == str:
        xml_tree = ET.parse(xml_path)
        xml_root = xml_tree.getroot()
        image_name_object = xml_root.find('filename')
        xml_root.find('filename').text = newfilename
        xml_tree.write(xml_path)

