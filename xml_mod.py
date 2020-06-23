import os
import sys
import xml.etree.ElementTree as ET
import glob

for file in glob.glob('*.xml'):
    file_name = file
    print(file_name)
    tree = ET.parse(file)
    root = tree.getroot()
    for elem in root.iter('name'):
        elem.text = 'S_carnosus'
    tree.write(file)
print("Done")