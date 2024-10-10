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

def update_image(new_data, im):
    im.set_data(new_data)
    fig.canvas.draw_idle()

    
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
    global image_name, save_count , file_list , current_image_index
    image_name = file_list[current_image_index]
    if os.path.isfile(image_name):
        print(f"Deleted {image_name}")
        os.remove(image_name)
        next_image(None)

# Configuration

target = 'cv2_diff_test / raw'

file_list = natsort.natsorted(glob(os.path.join(target , '*.jpg')))
file_len = len(file_list)

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
root.bind('<Return>', delete_image)
root.bind('<d>', next_image)
root.bind('<a>', previous_image)
root.bind('<Right>', next_image)
root.bind('<Left>', previous_image)
root.bind('<Delete>', delete_image)

root.mainloop()