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
    new_name = os.path.join(r"file path", os.path.basename(r) + "NIRv.tif")
    print(new_name)
    r = '\"' + r + '\"'
    n = '\"' + n + '\"'


    arcpy.gp.RasterCalculator_sa("(0.08-({n} - {r}) / ({n} + {r})) * {n}".format(n=n, r=r), new_name)

