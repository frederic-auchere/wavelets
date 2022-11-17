import time
import cProfile
from watroo import wow
from astropy.io import fits

image = fits.getdata('solo_L2_eui-hrieuv174-image_20220317T032000234_V01.fits')

#start = time.perf_counter()
start = time.time()
wow_image, _ = wow(image, bilateral=1, denoise_coefficients=[5, 2])
finish = time.time()
print(f"wow finished in {round(finish-start, 2)} ms")

fits.writeto('tata.fits', wow_image, overwrite=True)
