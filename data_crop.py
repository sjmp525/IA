'''
This code is aim to crop GID15 to 2048 * 2048

'''

from PIL import Image
import math
import os 

def crop_gid2_2048(image_path : str,mask_path : str,image_save_path : str,mask_save_path : str):
    img = Image.open(image_path)
    mask = Image.open(mask_path)
    fname = image_path.split('/')[-1].split('.')[0]
    idx = 0

    width,hight = img.size
    n_h,n_w = 0,0
    n_h = int(hight / 2048 + 1)
    n_w = int(width / 2048 + 1)
    for ih in range(n_h - 1):
        idx_h_s = ih * 2048
        idx_h_e = idx_h_s + 2048 
        for iw in range(n_w - 1):
            idx_w_s = iw * 2048
            idx_w_e = idx_w_s + 2048
            cropped_image = img.crop((idx_w_s, idx_h_s, idx_w_e, idx_h_e))
            # 保存裁剪后的图像
            cropped_image.save(os.path.join(image_save_path,fname+'_'+str(idx)+'.tif'))
            cropped_mask = mask.crop((idx_w_s, idx_h_s, idx_w_e, idx_h_e))
            # 保存裁剪后的图像
            cropped_mask.save(os.path.join(mask_save_path,fname+'_'+str(idx)+'.tif'))
            idx += 1
    for ih in range(n_h - 1):
        idx_h_s = ih * 2048
        idx_h_e = idx_h_s + 2048 
        idx_w_s = width - 2048
        idx_w_e = width
        cropped_image = img.crop((idx_w_s, idx_h_s, idx_w_e, idx_h_e))
        cropped_image.save(os.path.join(image_save_path,fname+'_'+str(idx)+'.tif'))
        cropped_mask = mask.crop((idx_w_s, idx_h_s, idx_w_e, idx_h_e))
        cropped_mask.save(os.path.join(mask_save_path,fname+'_'+str(idx)+'.tif'))
        idx += 1
    for iw in range(n_w - 1):
        idx_w_s = iw * 2048
        idx_w_e = idx_w_s + 2048 
        idx_h_s = hight - 2048
        idx_h_e = hight
        cropped_image = img.crop((idx_w_s, idx_h_s, idx_w_e, idx_h_e))
        cropped_image.save(os.path.join(image_save_path,fname+'_'+str(idx)+'.tif'))
        cropped_mask = mask.crop((idx_w_s, idx_h_s, idx_w_e, idx_h_e))
        cropped_mask.save(os.path.join(mask_save_path,fname+'_'+str(idx)+'.tif'))
        idx += 1
    idx_w_s = width - 2048
    idx_w_e = width
    idx_h_s = hight - 2048
    idx_h_e = hight
    cropped_image = img.crop((idx_w_s, idx_h_s, idx_w_e, idx_h_e))
    cropped_image.save(os.path.join(image_save_path,fname+'_'+str(idx)+'.tif'))
    cropped_mask = mask.crop((idx_w_s, idx_h_s, idx_w_e, idx_h_e))
    cropped_mask.save(os.path.join(mask_save_path,fname+'_'+str(idx)+'.tif'))
    idx += 1


if __name__ == '__main__':
    path = ''
    image_save_path = ''
    mask_save_path = ''
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split('.')[-1] == 'tif':
                file_path = os.path.join(root, file)
                label_path = file_path.replace('image','mask')
                directory, filename = os.path.split(label_path)
                base_name, extension = os.path.splitext(filename)
                new_filename = base_name + '_15label' + extension
                label_path = os.path.join(directory, new_filename)
                crop_gid2_2048(file_path,label_path,image_save_path,mask_save_path)