import os
import glob
from PIL import Image

files = glob.glob('images/*.JPG')
i = 0
for f in files:
    img = Image.open(f)
    img_resize = img.resize((int(img.width / 2), int(img.height / 2)))
    #title, ext = os.path.splitext(f)
    img_resize.save("reshaped/" + str(i) + ".jpg")
    i += 1