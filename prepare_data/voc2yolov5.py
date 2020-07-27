import os
import glob
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
# 处理voc格式数据


sets = ['train','test']
classes = ['car','arm_end','lighter']
def xml_to_csv(xml_path_list):
    xml_list = []
    for xml_file in xml_path_list:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def xml_to_text(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    return_text = ''
    for member in root.findall('object'):
        difficult = member.find('difficult').text
        cls = member.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = member.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        return_text+=str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n'
    return  return_text


def voc_text():
    root =r'/Data/jing/arm'
    if not os.path.exists(os.path.join(root, 'images')):
        os.mkdir(os.path.join(root, 'images'))
    if not os.path.exists(os.path.join(root, 'labels')):
        os.mkdir(os.path.join(root, 'labels'))
    for directory in ['train', 'val']:
        xml_dir = os.path.join(root, directory, 'annotations')
        if not os.path.exists(os.path.join(root,'images',directory)):
            os.mkdir(os.path.join(root,'images',directory))
        if not os.path.exists(os.path.join(root,'images',directory)):
            os.mkdir(os.path.join(root,'images',directory))
        if not os.path.exists(os.path.join(root,'labels',directory)):
            os.mkdir(os.path.join(root,'labels',directory))
        if not os.path.exists(os.path.join(root,'labels',directory)):
            os.mkdir(os.path.join(root,'labels',directory))
        xml_path_list = [os.path.join(xml_dir,i) for i in os.listdir(xml_dir)]
        for xml_path in xml_path_list:
            return_text= xml_to_text(xml_path)
            name = os.path.split(xml_path)[1].split('.')[0]
            with open(os.path.join(root,'labels',directory,name+'.txt'),'w') as f:
                f.write(return_text)
            shutil.copy(os.path.join(root,directory,'images',name+'.jpg'),
                        os.path.join(root, 'images', directory, name + '.jpg'))
    print('Successfully converted xml to text.')


def voc_xml():
    ## 训练数据路径
    root = r'C:\PythonProject\dataset\arm'
    for directory in ['train', 'val']:
        xml_dir = os.path.join(root, directory, 'annotations')
        xml_path_list = [os.path.join(xml_dir, i) for i in os.listdir(xml_dir)]
        xml_df = xml_to_csv(xml_path_list)
        xml_df.to_csv(os.path.join(root, '{}_labels.csv'.format(directory)), index=None)
        print('Successfully converted xml to csv.')


if __name__ == '__main__':
    voc_text()
