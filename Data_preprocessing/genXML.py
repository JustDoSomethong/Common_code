# -*- coding: UTF-8 -*-
import os
import cv2

def replace(line, src):
    p1 = line.find('>')
    p2 = line.find('<', p1)
    return line[:p1 + 1] + src + line[p2:]


def genXML(txt_folder, xml_folder, ref='./IVIPC_1.xml'):
    with open(ref, 'r') as f:
        ref = f.readlines()

    if not os.path.exists(xml_folder):
        os.mkdir(xml_folder)

    txt_list = os.listdir(txt_folder)

    for txt in txt_list:
        with open(os.path.join(txt_folder, txt), 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        for line in lines:
            if 'jpg' in line:
                filename = line
                name = filename[:-4]
                dst = os.path.join(xml_folder, name + '.xml')
                xml = open(dst, 'w')
                head = ref[:19]
                head[1] = replace(head[1], 'RHD')
                head[2] = replace(head[2], filename)
                head[11] = replace(head[11], 'KangYin')
                head[14] = replace(head[14], '320')
                head[15] = replace(head[15], '320')
                for ele in head:
                    xml.write(ele)
            else:
                pos = line.split(' ')
                obj = ref[19:31]
                pos = [int(p) for p in pos]
                if pos[4] == 3:
                    label = 'fist'
                else:
                    label = 'hand'
                pos = [str(p-1) for p in pos]

                obj[1] = replace(obj[1], label)
                obj[6] = replace(obj[6], pos[0])
                obj[7] = replace(obj[7], pos[1])
                obj[8] = replace(obj[8], pos[2])
                obj[9] = replace(obj[9], pos[3])
                for ele in obj:
                    xml.write(ele)
        xml.write('</annotation>')
        xml.close()


def genXML_new(txt_file, xml_folder, ref='./IVIPC_1.xml'):
    with open(ref, 'r') as f:
        ref = f.readlines()

    if not os.path.exists(xml_folder):
        os.mkdir(xml_folder)

    for line in open(txt_file, 'r'):

        lines = line.strip().split(' ')
        filename = lines[0]
        name = filename[:-4]
        label = lines[1]
        dst = os.path.join(xml_folder, name + '.xml')
        xml = open(dst, 'w')
        head = ref[:19]
        head[1] = replace(head[1], 'RHD')
        head[2] = replace(head[2], filename)
        head[11] = replace(head[11], 'KangYin')

        im = cv2.imread(os.path.join('/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/SSD_data/JPEGImages',filename))
        width = im.shape[1]
        height = im.shape[0]


        head[14] = replace(head[14], str(width))
        head[15] = replace(head[15], str(height))
        for ele in head:
            xml.write(ele)
        pos = lines[2:]
        obj = ref[19:31]
        obj[1] = replace(obj[1], label)
        obj[6] = replace(obj[6], pos[0])
        obj[7] = replace(obj[7], pos[1])
        obj[8] = replace(obj[8], pos[2])
        obj[9] = replace(obj[9], pos[3])
        for ele in obj:
            xml.write(ele)
        xml.write('</annotation>')
        xml.close()

if __name__ == '__main__':
    genXML_new('/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/SSD_data/shared_label.txt', '/media/pro/PRO/Ubuntu/Common_code/Data_preprocessing/SSD_data/xml')

