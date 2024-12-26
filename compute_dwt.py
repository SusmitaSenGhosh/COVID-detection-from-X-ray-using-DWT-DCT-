# import packages
import cv2
import numpy as np
import pywt
# %% set paths
txt_file_path = 'D:/Simpi/Work/COVID19/For_paper/txtfiles/'
preprocessed_image_path = 'C:/Users/susmi/for_paper/preprocessed_merged/all_images/'
dwt_output_path = 'C:/Users/susmi/for_paper/dwt/'
#%% defintion

def compute_dwt(data_dir, text_file, save_path,B = 8):
    line_content= text_file
    for i in range(0,len(line_content)):
        file_name = line_content[i].split(" ")[1]
        img = np.asarray(cv2.imread(data_dir + file_name),dtype = np.float32)/255
        h,w,ch= np.array(img.shape[:3])
        B = 112
        blocksV=int(h/B)
        blocksH=int(w/B)
        y = np.zeros((h,w,ch), np.float32)
        for channel in range(0,3):
            x = img[:,:,channel]
            coeffs2 = pywt.dwt2(x, 'haar')
            temp = [coeffs2[0], coeffs2[1][0], coeffs2[1][1], coeffs2[1][2]]
            count = 0
            for row in range(blocksV):
                    for col in range(blocksH):
                            y[row*B:(row+1)*B,col*B:(col+1)*B,channel]=temp[count]
                            count = count + 1
        np.save(save_path+'/'+file_name,y)

#%% main

with open(txt_file_path+'merged'+'.txt', 'r') as fr:
     all_files = fr.readlines()
     
compute_dwt(preprocessed_image_path,all_files,dwt_output_path, B = 8)
