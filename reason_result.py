import os
from glob import glob
import natsort


name = 'detect_scen4.1'
camera_number = '_camera6'

START = 1

reason_target = 13

box_target = reason_target - START + 1

All_file = natsort.natsorted(glob(os.path.join(f"{name}" , f"{camera_number}_image_raw" , "*.jpg")))
AI_detect_file = natsort.natsorted(glob(os.path.join(name ,f"{camera_number}_box_result",f"box{box_target}" , "*.jpg")))
print('AI_DETECT_FILE LEN :',len(AI_detect_file))

print("Reason Folder Number : ",reason_target)
User_find_file = natsort.natsorted(glob(os.path.join('inspect',f"{name}" , f"{camera_number}_image_raw" , f"{reason_target}" , "*.jpg")))
Not_detect_file = natsort.natsorted(glob(os.path.join('reason' , f"{name}" , f"{camera_number}" , f"{reason_target}" , str(1) , "*.jpg")))
Out_line_file = natsort.natsorted(glob(os.path.join('reason',f"{name}" ,f"{camera_number}" , f"{reason_target}" , str(2) , "*.jpg")))


if not Not_detect_file and not Out_line_file:
    print("EMPTY")
    exit(0)


print("=====================================================")
print("Camera : ",camera_number)
print("BUS SHEET NUMBER : ",reason_target)
print()
print("AIL file len :",len(All_file))
print("User find len :",len(User_find_file))
print()
print("AI 감지한 이미지 :",len(AI_detect_file))
print()
print("Detection Failed : ",len(Not_detect_file))
print("Out line :",len(Out_line_file))
print("=====================================================")