# -*- coding: utf-8 -*-
from osgeo import gdal
import os
from random import randint
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import sys
sys.path.append("..")
from SG_simple import simple_sg_filter as sgf
from curve_fitting import SMFS


#Importing images using the GDAL library
os.chdir(r"G:\DATA3\2020SMFs_data")
dts = gdal.Open("corn2020IL_mtr1.tif")
proj, geot = dts.GetProjection(), dts.GetGeoTransform() # get geographic coordinates
img_cover = dts.ReadAsArray() # get corn cover image
img_nirv_ts = gdal.Open("NIRv2020_IL").ReadAsArray() # get NIRv time series image

print(img_cover.shape)
print(img_nirv_ts.shape)


# The experimental area was displayed using matplotlib, and some of the maize NIRv curves were plotted
def random_a_point():
    while True:
        x, y = randint(0, 2657), randint(0, 3446)
        if img_cover[x, y] == 1:
            return x, y

ticks_font = FontProperties(fname="C:\\Windows\\Fonts\\arial.ttf", size=14)
DOYs = np.arange(1, 366, 8)
plt.figure(figsize=(9, 9))

plt.subplot(221)
plt.title("Corn Mapping")
plt.imshow(img_cover, vmin=0.1, vmax=0.6)
plt.xticks([], []); plt.yticks([], [])

plt.subplot(222)
plt.title("Mean NIRv")
plt.imshow(np.mean(img_nirv_ts, axis=0), cmap="jet")
plt.xticks([], []); plt.yticks([], [])
plt.colorbar()


time_labels_4 = ["Mar", "Jun", "Sep","Dec"]
time_xlength_4 = np.arange(45, 365, 90)


plt.subplot(223)
plt.title("Random Selected NIRv curve")
p = random_a_point()
print(p)
plt.plot(DOYs, img_nirv_ts[:, p[0], p[1]], lw=3, c='k')
plt.xticks(time_xlength_4, time_labels_4, fontproperties=ticks_font)
plt.yticks(fontproperties=ticks_font)
plt.xlabel("Month", fontproperties=ticks_font)
plt.ylabel("NIRv", fontproperties=ticks_font)


plt.subplot(224)
plt.title("Random Selected NIRv curve")
p = random_a_point()
print(p)
plt.plot(DOYs, img_nirv_ts[:, p[0], p[1]], lw=3, c='k')
plt.xticks(time_xlength_4, time_labels_4, fontproperties=ticks_font)
plt.yticks(fontproperties=ticks_font)
plt.xlabel("Month", fontproperties=ticks_font)
plt.ylabel("NIRv", fontproperties=ticks_font)
plt.show()


# Import reference NIRv and reference phenology
path_ref_nirv = open('NIRv2008shapemodle.txt')
nirv_ts_ref = [] #使用列表储存
for line in path_ref_nirv.readlines():
    nirv_ts_ref.append(float(line))
nirv_ts_ref = np.array(nirv_ts_ref)

path_ref_phe = open('ref_phe_2.txt')
phe_ref = []
for line in path_ref_phe.readlines():
    phe_ref.append(float(line))
phe_ref = np.array(phe_ref)

plt.figure(figsize=(5, 4))
plt.plot(DOYs, nirv_ts_ref, lw=3, c='k')
plt.xticks(time_xlength_4, time_labels_4, fontproperties=ticks_font)
plt.yticks(fontproperties=ticks_font)
plt.xlabel("Month", fontproperties=ticks_font)
plt.ylabel("NIRv", fontproperties=ticks_font)
for i in range(7):
    plt.scatter(phe_ref[i], np.interp(phe_ref[i], DOYs, nirv_ts_ref), c="C{}".format(i), zorder=10, edgecolor='k', s=70)
plt.show()

# Filtering 2D arrays
from tqdm import tqdm
img_nirv_ts2 = np.copy(img_nirv_ts)
xlen, ylen = img_cover.shape

for i in tqdm(range(xlen)):
    for j in range(ylen):
        #print(xlen, ylen)
        #print(i,j)
        img_nirv_ts2[:, i, j] = sgf(img_nirv_ts[:, i, j])

plt.figure(figsize=(9, 4))

plt.subplot(121)
p = random_a_point()
plt.plot(DOYs, img_nirv_ts[:, p[0], p[1]], lw=3, c="r", ls='--')
plt.plot(DOYs, img_nirv_ts2[:, p[0], p[1]], lw=3, c="k", ls='-')
# 设置坐标轴和标签
plt.xticks(time_xlength_4, time_labels_4, fontproperties=ticks_font)
plt.yticks(fontproperties=ticks_font)
plt.xlabel("Month", fontproperties=ticks_font)
plt.ylabel("NIRv", fontproperties=ticks_font)


plt.subplot(122)
p = random_a_point()
plt.plot(DOYs, img_nirv_ts[:, p[0], p[1]], lw=3, c="r", ls='--')
plt.plot(DOYs, img_nirv_ts2[:, p[0], p[1]], lw=3, c="k", ls='-')
plt.xticks(time_xlength_4, time_labels_4, fontproperties=ticks_font)
plt.yticks(fontproperties=ticks_font)
plt.xlabel("Month", fontproperties=ticks_font)
plt.ylabel("NIRv", fontproperties=ticks_font)

#plt.show()
plt.savefig(r"G:\DATA3\2009SMFs_data\randdompoint1SG.jpg", dpi=500)

# This code further processes the filtered array to make the curve flat at some specific points (smoothing out the curve at the ends of the growing season)
from scipy.signal import argrelextrema
BREAK_POINTS = (100, 320)
def flatten(curve, doy=DOYs):
    minimal_locs = np.array(argrelextrema(curve, np.less_equal)).reshape(-1)
    locs = np.interp(BREAK_POINTS, doy, np.arange(46))
    first_mini = minimal_locs[np.argmin(np.abs(minimal_locs - locs[0]))]
    second_mini = minimal_locs[np.argmin(np.abs(minimal_locs - locs[1]))]
    curve = np.copy(curve)
    curve[:first_mini] = curve[first_mini]
    curve[second_mini:] = curve[second_mini]
    return curve

img_nirv_ts3 = np.copy(img_nirv_ts2)
for i in range(xlen):
    for j in range(ylen):
        if img_cover[i, j]:
            img_nirv_ts3[:, i, j] = flatten(img_nirv_ts2[:, i, j])

nirv_fts_ref = flatten(nirv_ts_ref)

plt.figure(figsize=(9, 4))

plt.subplot(121)
p = random_a_point()
plt.plot(DOYs, img_nirv_ts[:, p[0], p[1]], lw=3, c="r", ls="--")
plt.plot(DOYs, img_nirv_ts2[:, p[0], p[1]], lw=3, c="k", ls="--")
plt.plot(DOYs, img_nirv_ts3[:, p[0], p[1]], lw=3, c="k", ls="-")
plt.xticks(time_xlength_4, time_labels_4, fontproperties=ticks_font)
plt.yticks(fontproperties=ticks_font)
plt.xlabel("Month", fontproperties=ticks_font)
plt.ylabel("NIRv", fontproperties=ticks_font)

plt.subplot(122)
p = random_a_point()
plt.plot(DOYs, img_nirv_ts[:, p[0], p[1]], lw=3, c="r", ls="--")
plt.plot(DOYs, img_nirv_ts2[:, p[0], p[1]], lw=3, c="k", ls="--")
plt.plot(DOYs, img_nirv_ts3[:, p[0], p[1]], lw=3, c="k", ls="-")
plt.xticks(time_xlength_4, time_labels_4, fontproperties=ticks_font)
plt.yticks(fontproperties=ticks_font)
plt.xlabel("Month", fontproperties=ticks_font)
plt.ylabel("NIRv", fontproperties=ticks_font)

#plt.show()
plt.savefig(r"G:\DATA3\2009SMFs_data\randdompointSG.jpg", dpi=500)

# Start using the SMF-S method to detect phenology
img_smfphes = np.zeros((len(phe_ref), xlen, ylen))
for phe_i in range(7):
    #print("开始监测", phe_i)
    smfs_model = SMFS(nirv_fts_ref, phe_ref[phe_i], DOYs)
    for i in tqdm(range(xlen)):
        for j in range(ylen):
            if not img_cover[i, j]:
                continue
            img_smfphes[phe_i, i, j] = smfs_model.runit(img_nirv_ts3[:, i, j])


# Randomly selected points to see the performance of climate monitoring
plt.figure(figsize=(16, 4))
for i_map in range(3):
    plt.subplot(1, 3, i_map + 1)
    p = random_a_point()
    p[0], p[1]
    plt.plot(DOYs, img_nirv_ts[:, p[0], p[1]], lw=3, c="r", ls="--")
    plt.plot(DOYs, img_nirv_ts2[:, p[0], p[1]], lw=3, c="k", ls="--")
    plt.plot(DOYs, img_nirv_ts3[:, p[0], p[1]], lw=3, c="k", ls="-")
    phe_detected = img_smfphes[:, p[0], p[1]]
    for i in range(7):
        if phe_detected[i] == 0:
            continue
        plt.scatter(phe_detected[i],
                    np.interp(phe_detected[i], DOYs, img_nirv_ts3[:, p[0], p[1]]),
                    c='C{}'.format(i),
                    s=70,
                    zorder=10,
                    edgecolor='k')
    plt.xticks(time_xlength_4, time_labels_4, fontproperties=ticks_font)
    plt.yticks(fontproperties=ticks_font)
    plt.xlabel("Month", fontproperties=ticks_font)
    plt.ylabel("NIRv", fontproperties=ticks_font)
#plt.show()
plt.savefig(r"G:\DATA3\2009SMFs_data\randomepoint_phe.jpg", dpi=500)
print("randomepoint_img are saved")

# Regional climatic mapping results can also be printed

plt.figure(figsize=(16, 12))
plt.subplot(331)
plt.title("Planted")
plt.imshow(img_smfphes[0, :, :], vmin=110, vmax=140, cmap="jet")
plt.xticks([], []); plt.yticks([], [])
plt.colorbar()

plt.subplot(332)
plt.title("Emerged")
plt.imshow(img_smfphes[1, :, :], vmin=130, vmax=150, cmap="jet")
plt.xticks([], []); plt.yticks([], [])
plt.colorbar()

plt.subplot(333)
plt.title("Silking")
plt.imshow(img_smfphes[2, :, :], vmin=190, vmax=210, cmap="jet")
plt.xticks([], []); plt.yticks([], [])
plt.colorbar()

plt.subplot(334)
plt.title("Dough")
plt.imshow(img_smfphes[3, :, :], vmin=220, vmax=240, cmap="jet")
plt.xticks([], []); plt.yticks([], [])
plt.colorbar()

plt.subplot(335)
plt.title("Dented")
plt.imshow(img_smfphes[4, :, :], vmin=230, vmax=260, cmap="jet")
plt.xticks([], []); plt.yticks([], [])
plt.colorbar()

plt.subplot(336)
plt.title("Mature")
plt.imshow(img_smfphes[5, :, :], vmin=250, vmax=280, cmap="jet")
plt.xticks([], []); plt.yticks([], [])
plt.colorbar()

plt.subplot(337)
plt.title("Harvested")
plt.imshow(img_smfphes[6, :, :], vmin=290, vmax=320, cmap="jet")
plt.xticks([], []); plt.yticks([], [])
plt.colorbar()

plt.show()

# Simply save the detection image using the GDAL library, saving the img_smfsphes array as an image file in GeoTIFF format
img_smfphes = img_smfphes.astype("uint16")
gdal_type = 2
im_band, im_height, im_width = img_smfphes.shape
driver = gdal.GetDriverByName("GTiff")
dataset = driver.Create("Img_NIRv_corn2020IL_phe.tif", im_width, im_height, im_band, gdal_type)
dataset.SetGeoTransform(geot)
dataset.SetProjection(proj)
for i in range(im_band):
    dataset.GetRasterBand(i + 1).WriteArray(img_smfphes[i, :, :])
dataset = None
