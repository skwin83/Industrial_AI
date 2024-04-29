import os
import cv2

obj = "can"

dic = obj + "_original/"

# Check file list
num_file = len(os.walk(dic).next()[2])/2
print num_file

num = 1


for file in range(num_file):
    txt_dict = obj+"_original/" + str(file+1) + ".txt"
    print txt_dict
    # read .txt file
    f = open(txt_dict,'r')
    data = f.read()
    cordinate = data.split()
    f.close
    
    # Yolo mark equation

    #float const relative_center_x = (float)(i.abs_rect.x + i.abs_rect.width / 2) / full_image_roi.cols;
    #float const relative_center_y = (float)(i.abs_rect.y + i.abs_rect.height / 2) / full_image_roi.rows;
    #float const relative_width = (float)i.abs_rect.width / full_image_roi.cols;
    #float const relative_height = (float)i.abs_rect.height / full_image_roi.rows;

    pic_dict = obj + "_original/" + str(file+1) + ".jpg"

    src = cv2.imread(pic_dict,cv2.IMREAD_COLOR)    
    dst = src.copy()
    
    x1 = int(float(cordinate[1])*1280 - float(cordinate[3])*1280/2)
    x2 = int(x1 + float(cordinate[3])*1280) 

    y1 = int(float(cordinate[2])*720 - float(cordinate[4])*720/2)
    y2 = int(y1 + float(cordinate[4])*720)

    dst = src[y1:y2,x1:x2]

    if not os.path.exists(obj):
        os.makedirs(obj)
    
    new_dict = obj + "/" + str(file+num) + ".jpg"
    cv2.imwrite(new_dict,dst)

    if len(cordinate) > 5:
        num = num + 1
        x1 = int(float(cordinate[6])*1280 - float(cordinate[8])*1280/2)
        x2 = int(x1 + float(cordinate[8])*1280) 

        y1 = int(float(cordinate[7])*720 - float(cordinate[9])*720/2)
        y2 = int(y1 + float(cordinate[7])*720)

        dst = src[y1:y2,x1:x2]
        new_dict = obj + "/" + str(file+num) + ".jpg"
        cv2.imwrite(new_dict,dst)
    

# Test
'''
txt_dict = obj+"_original/" + str(1) + ".txt"
print txt_dict
# read .txt file
f = open(txt_dict,'r')
data = f.read()
cordinate = data.split()
f.close

pic_dict = obj + "_original/" + str(1) + ".jpg"

src = cv2.imread(pic_dict,cv2.IMREAD_COLOR)       
dst = src.copy()

x1 = int(float(cordinate[6])*1280 - float(cordinate[8])*1280/2)
x2 = int(x1 + float(cordinate[8])*1280) 

y1 = int(float(cordinate[7])*720 - float(cordinate[9])*720/2)
y2 = int(y1 + float(cordinate[7])*720)

dst = src[y1:y2,x1:x2]

cv2.imshow("src",src)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destoryAllWindows()
'''