import time
import cProfile
from watroo import wow
from astropy.io import fits
import glob
import numpy as np

files_list=sorted(glob.glob("/run/media/cmercier/LaCie/eui_data/20220402/*.fits"))

small_file_list = files_list[0:450]

for i, f in enumerate(small_file_list):
    #print (f)
    image = fits.getdata(f)
    image = image.astype('float32')
    if i == 0:
        print (image.shape)
        print ('truc ', (len(small_file_list),), ' ddddddddd')
        cube = np.empty_like(image, shape= (len(small_file_list),)  + image.shape)
    cube[i] = image

cube = cube[:, 0:1024, 0:1024] 

#start = time.perf_counter()
start = time.time()
wow_image, _ = wow(cube, bilateral=1, denoise_coefficients=[5, 2])
finish = time.time()
print(f"wow finished in {round(finish-start, 2)} s")

fits.writeto('3Dtata.fits', wow_image, overwrite=True)
