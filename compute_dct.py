# import packages
import cv2
import numpy as np
# %% set paths
txt_file_path = 'D:/Simpi/Work/COVID19/For_paper/txtfiles/'
preprocessed_image_path = 'C:/Users/susmi/for_paper/preprocessed_merged/all_images/'
dct_output_path = 'C:/Users/susmi/for_paper/dct/'

#%% defintion

def compute_dct(data_dir, text_file, save_path,B = 8):
    line_content= text_file
    for i in range(0,len(line_content)):
        file_name = line_content[i].split(" ")[1]
        img = np.asarray(cv2.imread(data_dir + file_name),dtype = np.float32)/255
        h,w,ch= np.array(img.shape[:3])
        blocksV=int(h/B)
        blocksH=int(w/B)
        Trans = np.zeros((h,w,ch), np.float32)
        for channel in range(0,3):
            x = img[:,:,channel]
            vis0 = np.zeros((h,w), np.float32)
            vis0[:h, :w] = x
            for row in range(blocksV):
                    for col in range(blocksH):
                            currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
                            Trans[row*B:(row+1)*B,col*B:(col+1)*B,channel]=currentblock
        np.save(save_path+'/'+file_name,Trans)

#%% main

with open(txt_file_path+'merged'+'.txt', 'r') as fr:
     all_files = fr.readlines()
     
compute_dct(preprocessed_image_path,all_files,dct_output_path, B = 8)
