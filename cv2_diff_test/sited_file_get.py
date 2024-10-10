import cv2
from glob import glob
import os
import matplotlib.pyplot as plt
import natsort
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import gc


file_count = 0

omit_count = 0
def load_image(image_path):
    global file_count
    if image_path in image_cache:
        image = image_cache[image_path]
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise FileExistsError(image_path)
        image_cache[image_path] = image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.cla()
    ax.imshow(image)
    ax.axis("off")
    image_name = os.path.basename(image_path)
    print("Load Image: {} ".format(image_name))
    ax.set_title(f"{image_path}")

    return image


# 다음 이미지로 이동
def next_image(event):
    global current_image_index, file_list, file_count
    file_count += 1
    if file_list:  # 파일 리스트가 비어 있지 않은 경우에만 다음 이미지로 이동
        current_image_index = (current_image_index + 1) % len(file_list)
        load_image(file_list[current_image_index])
        canvas.draw_idle()
        gc.collect()


# 이전 이미지로 이동
def previous_image(event):
    global current_image_index, file_list, file_count
    file_count -= 1
    if file_list:  # 파일 리스트가 비어 있지 않은 경우에만 이전 이미지로 이동
        current_image_index = (current_image_index - 1) % len(file_list)
        load_image(file_list[current_image_index])
        canvas.draw_idle()
        gc.collect()


# 이미지 삭제
def delete_image(event):
    global current_image_index, file_list, file_count
    file_count -= 1
    if file_list:  # 파일 리스트가 비어 있지 않은 경우에만 삭제 시도
        image_path = file_list[current_image_index]
        # 파일이 존재하는지 확인 후 삭제
        if os.path.isfile(image_path):
            os.remove(image_path)
            print(f"Deleted: {image_path}")

        # 캐시에서 이미지 삭제
        image_cache.pop(image_path, None)

        # 파일 리스트에서 제거
        file_list.remove(image_path)

        # try:
        #     test = cv2.imread(image_path)
        #     if test is not None or test:
        #         raise FileExistsError(image_path)
        # except:
        #     pass
        # 삭제 후 리스트가 비어있는지 확인

        if os.path.isfile(image_path):
            raise FileExistsError(image_path)
        if not file_list:
            ax.cla()
            ax.set_title("No images found.")
        else:
            # 인덱스 조정 (범위를 벗어나지 않도록 처리)
            current_image_index = current_image_index % len(file_list)
            load_image(file_list[current_image_index])

        canvas.draw_idle()
        gc.collect()

def save_image(event):
    global current_image_index, file_list, file_count , omit_count
    omit_count +=1
    
    SCEN = file_list[current_image_index].split('/')[-2]
    image_name = file_list[current_image_index].split('/')[-1]
    mat = cv2.imread(file_list[current_image_index] , cv2.IMREAD_COLOR)
    print("SAVE IMAGE : ",image_name)
    
    full_name = SCEN + "_" + image_name
    save_loc = os.path.join(SAVE_FOLDER , full_name)
    if not os.path.isfile(save_loc):
        cv2.imwrite(save_loc ,  mat)
    else:
        print("Already Saved")
    next_image(None)

def delete_image(event):
    global current_image_index, file_list, file_count , omit_count
    omit_count -=1
    SCEN = file_list[current_image_index].split('/')[-2]
    T = file_list[current_image_index]
    name = SCEN + '_' + file_list[current_image_index].split('/')[-1]
    path = os.path.join(SAVE_FOLDER , name)
    if os.path.isfile(path):
        os.remove(path)
        print("Deleted : ",name)
    else:
        print("Dose not Exist : ",path)
        next_image(None)

    del image_cache[file_list[current_image_index]]
    file_list.remove(file_list[current_image_index])
    next_image(None)

# 이미지 캐시
image_cache = {}

base_folder = "cv2_diff_test/front"
file_list = natsort.natsorted(
    glob(os.path.join(base_folder, "**", "*.jpg"), recursive=True)
)
Number = 13
SAVE_FOLDER = f'cv2_diff_test/raw_seat/{Number}'
os.makedirs(SAVE_FOLDER , exist_ok=True)

len_file_list = len(file_list)
print("file len :", len(file_list))
current_image_index = 0

root = tk.Tk()
root.geometry("600x600+1000+200")
plt.ioff()
fig, ax = plt.subplots(dpi=80)

if file_list:  # 파일 리스트가 비어 있지 않은 경우에만 초기 이미지 로드
    image = load_image(file_list[current_image_index])
else:
    ax.set_title("No images found.")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

root.bind("<d>", next_image)
root.bind("<a>", previous_image)
root.bind("<Right>", next_image)
root.bind("<Left>", previous_image)
root.bind("<s>" , save_image)
root.bind("<S>" , save_image)
root.bind("<Delete>", delete_image)

root.mainloop()
