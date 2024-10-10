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
current_image_index = 0

name = 'detect_scen4.1'
image_cache = {}
image_name = None
mode = 'reason'
text_handle = None
iter_count = 0

camera_focues = '_camera8'



target_sheet = 4


START = 1
standard = START + target_sheet - 1 
print("Real Target Sheet : ",standard)

#pdb.set_trace()
ai_path = os.listdir(f'{name}/{camera_focues}_box_result/box{target_sheet}')
print("===========TARGET :{} ===============".format(target_sheet))
print("AI DETECT IMAGE LEN :", len(ai_path))
try:
    user_path = os.listdir(f'inspect/{name}/{camera_focues}_image_raw/{standard}')
    print("User DETECT IMAGE LEN :", len(user_path))
except:
    print("PATH : ",f'inspect/{name}/{camera_focues}_image_raw/{standard}')
    print("========user path not image=======")
    exit(0)

def wrapper(event):
    save_image(event, image)

def update_image(new_data, im):
    im.set_data(new_data)
    fig.canvas.draw_idle()

def save_image(event, image):
    new_folder_1 = f"{mode}/{name}/{camera_focues}/{standard}/1"
    new_folder_2 = f"{mode}/{name}/{camera_focues}/{standard}/2"
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
    image_path_ = os.path.join(f'{mode}', name, str(target_sheet), str(1), image_name)
    image_path__ = os.path.join(f'{mode}', name, str(target_sheet), str(2), image_name)
    if os.path.isfile(image_path_):
        print(f"Deleted Reason 1 {image_name}")
        os.remove(image_path_)
        
    if os.path.isfile(image_path__):
        print(f"Deleted Reason 2 {image_name}")
        os.remove(image_path__)

diff_path = set(user_path) - set(ai_path)
# test_path  = set(ai_path) - set(user_path)
print("AI 못찾은 개수 : ",len(diff_path))

file_list = natsort.natsorted([os.path.join(f'{name}/{camera_focues}_image_raw',i) for i in diff_path])
file_len = len(file_list)
if file_len == 0:
    exit(0)
    
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