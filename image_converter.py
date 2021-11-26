import shutil
from pathlib import Path
DIR = 'trainlms'
Path(DIR + '/cat').mkdir(exist_ok=True)
Path(DIR + '/dog').mkdir(exist_ok=True)


files = Path(DIR).glob('*.jpg')
for file in files:
    file_name = str(file)
    image_name = file_name.split('\\')[1]
    _class = image_name.split('.')[0]
    shutil.move(file_name, DIR+f"/{_class}/{image_name[4:]}")


