# import ftplib
# import os

# import matplotlib.pyplot as plt
# import numpy as np
# from astropy.io import fits
# from matplotlib.ticker import FuncFormatter

# y = [1,2,3,4,5,6,7,8,9]
# x = [1,2,None,4,5,6,7,8,9]

# plt.plot(x,y)
# plt.show()


from casatools import image
im = image()
im.open('5800.image/')
im.open('11800_1.image/')
beam_params = im.restoringbeam()
print(beam_params)

