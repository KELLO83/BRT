import cv2
from glob import glob
import os
import matplotlib.pyplot as plt
import natsort
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm.auto import tqdm
import matplotlib
import gc
import pdb
os.environ["QT_DEBUG_PLUGINS"] = "0"

user_path_file_name = []
ai_path_file_name = []
do_not_find_path = []

name = '33'

user_path = glob(os.path.join(f'group/{name}',"*.jpg"))
ai_box_path = glob(os.path.join(f'group/box_dir/{name}/box1',"*.jpg"))
print("user path file len :",len(user_path))
print("ai path file len :",len(ai_box_path))



for i in user_path:
    user_path_file_name.append(os.path.basename(i))
    
for k in ai_box_path:
    ai_path_file_name.append(os.path.basename(k))
    

diff_path = set(user_path_file_name) - set(ai_path_file_name)
diff_path = natsort.natsorted(diff_path)

for j in diff_path:
    target = os.path.join(f'group/{name}',j)
    do_not_find_path.append(target)


file_list = do_not_find_path.copy()
print('==========================================================')
print("TARGET : ", name)
print("INSPECTION FILE len :",len(diff_path))
print('==========================================================')
flag = str(input("Continue Any key press"))
if flag.upper == 'N':
    exit(0)
    
image_cache = {}
mode = 'reason'
current_image_index = 0
text_handle  = None
iter_count = 0 

location = [((480,502) , (749,604))]

file_len = len(file_list)

def wrapper(event):
    save_image(event, image)

def save_image(event, image):
    new_folder_1 = f"reason/{name}/1"
    new_folder_2 = f"reason/{name}/2"
    os.makedirs(new_folder_1, exist_ok=True)
    os.makedirs(new_folder_2, exist_ok=True)
             
    if event.keysym == 's':  
        image_path = os.path.join(new_folder_1, os.path.basename(file_list[current_image_index]))
        compare_path = os.path.join(new_folder_2 , os.path.basename(file_list[current_image_index]))
        if not os.path.isfile(image_path) and not os.path.isfile(compare_path):
            print(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_path, image)
            next_image(None)

        else:
            print(f"{os.path.basename(file_list[current_image_index])} is aleary saved")
            try:
                os.remove(image_path)
            except:
                pass
            try:
                os.remove(compare_path)
            except:
                pass
            print(image_path)
            cv2.imwrite(image_path , image)
            next_image(None)
            
    elif event.keysym =='d':
        image_path = os.path.join(new_folder_2 , os.path.basename(file_list[current_image_index]))
        compare_path = os.path.join(new_folder_1 , os.path.basename(file_list[current_image_index]))
        if not os.path.isfile(image_path) and not os.path.isfile(compare_path):
            print(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_path, image)
            next_image(None)
            
        else:
            print(f"{os.path.basename(file_list[current_image_index])} is aleary saved")
            try:
                os.remove(image_path)
            except:
                pass
            
            try:
                os.remove(compare_path)
            except:
                pass
            print(image_path)
            cv2.imwrite(image_path , image)
            next_image(None)
            
    else:
        raise KeyboardInterrupt(f"{event.keysym} key Error")
    
def load_image(image_path):
    global text_handle, file_len, iter_count, text_file_count, image_name
    if image_path in image_cache:
        image = image_cache[image_path]
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileExistsError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for (x1 ,y1) ,(x2,y2) in location:
            cv2.rectangle(image , (x1,y1) , (x2,y2),color=(0,255,0),thickness=3)
        image_cache[image_path] = image

    ax.cla()
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    image_name = os.path.basename(image_path)
    ax.set_title(f"{os.path.basename(image_path)}")
    if text_handle:
        text_handle.remove()
        text_file_count.remove()
    file_count = (iter_count % file_len) + 1
    text_handle = ax.text(0.5, 0.5, f"not detect -> s", fontsize=15, ha='center', color='r')
    text_file_count = ax.text(500, 0.5, f"{file_count} / {file_len}", fontsize=15, ha='center', color='b')
    iter_count += 1

    return image


def next_image(event):
    global current_image_index, image, text_file_count
    current_image_index = (current_image_index + 1) % len(file_list)
    image = load_image(file_list[current_image_index])
    canvas.draw_idle()
    gc.collect()

def previous_image(event):
    global current_image_index, image, text_file_count, iter_count
    iter_count -= 2
    current_image_index = (current_image_index - 1) % len(file_list)
    image = load_image(file_list[current_image_index])
    canvas.draw_idle()
    gc.collect()
    
def delete_image(event):
    global image_name
    image_path_ = os.path.join(f'reason/{name}',str(1), image_name)
    image_path__ = os.path.join(f'reason/{name}',str(2), image_name)
    if os.path.isfile(image_path_):
        print(f"Deleted Reason 1 {image_name}")
        os.remove(image_path_)
        
    if os.path.isfile(image_path__):
        print(f"Deleted Reason 2 {image_name}")
        os.remove(image_path__)

root = tk.Tk()
root.title("Image plot")
root.geometry('600x600+1000+200')

#matplotlib.use('TkAgg')
plt.ioff()
fig, ax = plt.subplots(dpi=80)

image = load_image(file_list[current_image_index])

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


root.bind('<s>',wrapper) 
root.bind('<d>',wrapper)
root.bind('<Right>', next_image)    
root.bind('<Left>',previous_image)
root.bind('<Delete>', delete_image)  

root.mainloop()
