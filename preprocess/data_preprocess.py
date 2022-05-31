import easyocr
import os

import glob
reader = easyocr.Reader(['en']) # need to run only once to load model into memory
dir_f = '/home/rosen/Project/img/EasyOCR/black'
from pathlib import Path

def show_txt(dir_f):
    for img in os.listdir(dir_f):
        print("start {}".format(img))
        txts = []
        img_p = os.path.join(dir_f, img)
        result = reader.readtext(img_p)
        for all in result:
            bbx = all[0]
            txt = all[1]
            prb = all[2]
            txts.append(txt)
        print(txts)


def show_txt_img(img):
    result = reader.readtext(img)
    print(result)


def clean_fine_grain(fg_dir, fg_dir_clean):
    """
    Clean images which doesn't includes signs.
    :param fg_dir:
    :param fg_dir_clean:
    :return:
    """
    # got reader
    reader = easyocr.Reader(['en']) # need to run only once to load model into memory

    for r, s, f in os.walk(fg_dir):
        if len(s) > 1 and len(f) == 0:  # first root
            pass
        if len(s) == 1 and len(f) == 0:  # second root
            pass
        if len(s) == 0 and len(f) > 0:  # last layer
            sec_dir = os.path.dirname(r)
            new_dir = os.path.join(fg_dir_clean, os.path.basename(r))
            Path(new_dir).mkdir(parents=True, exist_ok=True)
            for img in f:
                org_img_p = os.path.join(r, img)
                try:
                    result = reader.readtext(org_img_p)
                except:
                    print(org_img_p)
                if len(result) == 0:
                    print("img: {} has no  sign.".format(org_img_p))
                    continue
                else:
                    print("cp {} to {}".format(org_img_p, new_dir))
                    os.system("cp {} {}".format(org_img_p, new_dir))


if __name__ == '__main__':
    img = dir_f + "/129.jpg"
    #show_txt_img(img)
    #dir_f = '/home/Data/image_data/fine_grain_img'
    #new_dir = "/home/Data/image_data/fine_grain_img_clean"

    # Presidential
    dir_f = '/home/Data/image_data/Presidential'
    new_dir = "/home/Data/image_data/Presidential_clean"
    clean_fine_grain(dir_f, new_dir)