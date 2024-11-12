from pathlib import Path

import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
from xrtpy.response.temperature_from_filter_ratio import \
    temperature_from_filter_ratio

result = Fido.search(
    a.Time("2023-03-20 07:27:09", "2023-03-20 07:40:00"), a.Instrument("xrt")
)

print(result)

data_files = Fido.fetch(result, progress=False)

file1 = data_files[1]
file2 = data_files[0]
print("Files used:\n", file1, "\n", file2)

map1 = sunpy.map.Map(file1)
map2 = sunpy.map.Map(file2)

print(
    map1.fits_header["TELESCOP"],
    map1.fits_header["INSTRUME"],
)
print(
    "\n File 1 used:\n",
    file1,
    "\n Observation date:",
    map1.fits_header["DATE_OBS"],
    map1.fits_header["TIMESYS"],
    "\n Filter Wheel 1:",
    map1.fits_header["EC_FW1_"],
    map1.fits_header["EC_FW1"],
    "\n Filter Wheel 2:",
    map1.fits_header["EC_FW2_"],
    map1.fits_header["EC_FW2"],
    "\n Dimension:",
    map1.fits_header["NAXIS1"],
    map1.fits_header["NAXIS1"],
)

print(
    "\nFile 2 used:\n",
    file2,
    "\n Observation date:",
    map2.fits_header["DATE_OBS"],
    map2.fits_header["TIMESYS"],
    "\n Filter Wheel 1:",
    map2.fits_header["EC_FW1_"],
    map2.fits_header["EC_FW1"],
    "\n Filter Wheel 2:",
    map2.fits_header["EC_FW2_"],
    map2.fits_header["EC_FW2"],
    "\n Dimension:",
    map2.fits_header["NAXIS1"],
    map2.fits_header["NAXIS1"],
)

T_EM = temperature_from_filter_ratio(map1, map2)
T_e = T_EM.Tmap

import matplotlib.pyplot as plt
import numpy as np
from sunpy.coordinates.sun import B0, angular_radius
from sunpy.map import Map

# To avoid error messages from sunpy we add metadata to the header:
rsun_ref = 6.95700e08
hdr1 = map1.meta
rsun_obs = angular_radius(hdr1["DATE_OBS"]).value
dsun = rsun_ref / np.sin(rsun_obs * np.pi / 6.48e5)
solarb0 = B0(hdr1["DATE_OBS"]).value
hdr1["DSUN_OBS"] = dsun
hdr1["RSUN_REF"] = rsun_ref
hdr1["RSUN_OBS"] = rsun_obs
hdr1["SOLAR_B0"] = solarb0

fig = plt.figure()
# We could create a plot simply by doing T_e.plot(), but here we choose to make a linear plot of T_e
m = Map((10.0**T_e.data, T_e.meta))
m.plot(title="Derived Temperature", vmin=2.0e6, vmax=1.2e7, cmap="turbo")
m.draw_limb()
m.draw_grid(linewidth=2)
cb = plt.colorbar(label="T (K)")
plt.show()