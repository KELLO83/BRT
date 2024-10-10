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
import shutil

# Global variables
save_count = 0
current_image_index = 0
text_handle = None
text_file_count = None
iter_count = 0
image_cache = {}
image_name = None

def wrapper(event):
    save_image(event, image)
    
def update_image(new_data, im):
    im.set_data(new_data)
    fig.canvas.draw_idle()

def save_image(event, image):
    global save_count
    new_folder = f"inspect/{name}/{target_sheet}"
    os.makedirs(new_folder, exist_ok=True)
    image_path = os.path.join(new_folder, os.path.basename(file_list[current_image_index]))

    if os.path.isfile(image_path):
        print(f"{image_path} is already saved")
    if not os.path.isfile(image_path):
        print(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path, image)
        save_count += 1
    previous_image(None)
    next_image(None) 
    
def load_image(image_path):
    global text_handle, file_len, iter_count, text_file_count, image_name, save_count
    if image_path in image_cache:
        image = image_cache[image_path]
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
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
    text_handle = ax.text(0.5, 0.5, f"{save_count}", fontsize=15, ha='center', color='r')
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
    global image_name, save_count
    image_path = os.path.join('inspect', name, str(target_sheet), image_name)
    if os.path.isfile(image_path):
        print(f"Deleted {image_name}")
        os.remove(image_path)
    save_count = len(os.listdir(os.path.join('inspect', name, str(target_sheet))))

# Configuration
name = 'detect_1-1/_camera6_image_raw'  # Folder name
target_sheet = 17  # Target seat number

file_list = natsort.natsorted(glob(os.path.join(name, '*.jpg')))
file_len = len(file_list)



# Starting point
try:
    save_count = len(os.listdir(os.path.join('inspect', name, str(target_sheet))))
except:
    save_count = 0

# target_dir = os.path.join('inspect', name, str(target_sheet))
# os.makedirs(os.path.join(target_dir) , exist_ok= True)
# for i in file_list:
#     shutil.copy(i,os.path.join(target_dir , os.path.basename(i)))
# else:
#     exit(0)
    

# Tkinter setup
root = tk.Tk()
root.title("Image plot")
root.geometry('800x800+1000+200')

matplotlib.use('TkAgg')
plt.ioff()
fig, ax = plt.subplots(dpi=80)

image = load_image(file_list[current_image_index])

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Event bindings
root.bind('<Return>', wrapper)
root.bind('<d>', next_image)
root.bind('<a>', previous_image)
root.bind('<Right>', next_image)
root.bind('<Left>', previous_image)
root.bind('<Delete>', delete_image)

root.mainloop()