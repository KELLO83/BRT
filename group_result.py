import os
from glob import glob
import natsort


sheet_number = 33

user_find_file = glob(os.path.join(f'group/raw/{sheet_number}' , '*.jpg'))
ai_find_file = glob(os.path.join(f'group/box_dir/{sheet_number}/box1' , '*.jpg'))
not_dtect_file = glob(os.path.join('reason',f'{sheet_number}',str(1) , '*.jpg'))
out_file = glob(os.path.join('reason' , f'{sheet_number}' , str(2) , '*.jpg'))

print("=====================================================")
print("BUS SHEET NUMBER : ", sheet_number)
print()
print("User find len :",len(user_find_file))
print()
print("AI 감지한 이미지 :",len(user_find_file) - len(out_file) - len(not_dtect_file))
print()
print("Out line :",len(out_file))
print("Detection Failed : ",len(not_dtect_file))
print()
print("=====================================================")