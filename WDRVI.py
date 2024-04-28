# -*- coding: utf-8 -*-
import glob
import os.path
import arcpy

# Local variables:
RED = "file path"
NIR = "file path"

red = glob.glob(RED)
nir = glob.glob(NIR)


print(red)
print(nir)

for r, n in zip(red, nir):
    print(r, n)
    # 定义生成文件名
    new_name = os.path.join(r"file path", os.path.basename(r) + "WDRVI.tif")
    print(new_name)
    r = '\"' + r + '\"'
    n = '\"' + n + '\"'

    # 在这里执行栅格计算
    arcpy.gp.RasterCalculator_sa("(0.1 * {n} - {r}) / (0.1 * {n} + {r})".format(n=n, r=r), new_name)


