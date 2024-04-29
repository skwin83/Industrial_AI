# Yolo mark
https://github.com/AlexeyAB/Yolo_mark<br>

- x와 y 좌표 얻는 법

해당 코드 위치:<br>
https://github.com/AlexeyAB/Yolo_mark/blob/ea049f3f1a4812500526db409e61e34c7bdcd4da/main.cpp#L530-L533

	float const relative_center_x = (float)(i.abs_rect.x + i.abs_rect.width / 2) / full_image_roi.cols;
	float const relative_center_y = (float)(i.abs_rect.y + i.abs_rect.height / 2) / full_image_roi.rows;
	float const relative_width = (float)i.abs_rect.width / full_image_roi.cols;
	float const relative_height = (float)i.abs_rect.height / full_image_roi.rows;

<br>

> x 좌표 : relative_center_x * full_image_roi.cols - i.abs_rect.width / 2

> y 좌표 : relative_center_y * full_image_roi.rows - i.abs_rect.height / 2

> 너비 : relative_width * full_image_roi.cols

> 높이 : relative_height * full_image_roi.rows

# cut_image.py
<br>
여기서 이미지 크기는 1280 x 720이다.<br> 
cordinate[1] = relative_center_x<br>
cordinate[2] = relative_center_y<br>
cordinate[3] = relative_width<br>
cordinate[4] = relative_height<br>

    x1 = int(float(cordinate[1])*1280 - float(cordinate[3])*1280/2)
    x2 = int(x1 + float(cordinate[3])*1280) 

    y1 = int(float(cordinate[2])*720 - float(cordinate[4])*720/2)
    y2 = int(y1 + float(cordinate[4])*720)

    dst = src[y1:y2,x1:x2] #[높이,너비]
