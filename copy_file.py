import os
from glob import glob
import natsort 
import shutil
from tqdm import tqdm
name = 'detect_scen6'
camera = '_camera8'

number = 29
target = natsort.natsorted(glob(os.path.join(f'{name}/{camera}_image_raw','*.jpg')))
dest  = f'inspect/{name}/{camera}_image_raw/{number}'

print(dest)
os.makedirs(dest , exist_ok=True)


for i in tqdm(target):
    shutil.copy(i , dest)