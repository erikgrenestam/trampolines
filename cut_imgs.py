import fiona
from shapely.geometry import shape, box, Point
from shapely.affinity import rotate
import os, os.path
from PIL import Image
from osgeo import gdal
import csv
import time

#@profile
def toJpeg(im,filename,max_side_dim,outFolder,angle):
    size = (max_side_dim, max_side_dim)
    if angle == 0:
        background = Image.new('RGB', size, (0, 0, 0)) 
        background.paste(im, (int((size[0] - im.size[0]) / 2), int((size[1] - im.size[1]) / 2)))
        if background.getbbox():
            background.save(outFolder+filename+'.jpg', 'JPEG', quality = 95)
            print("Jpeg exported.")
            return 1
        else:
            print("All black!")
            return 2
    else:
        w, h = im.size
        size_large = (max_side_dim*4, max_side_dim*4)
        background = Image.new('RGB', size_large, (0, 0, 0)) 
        background.paste(im, (int((size_large[0] - im.size[0]) / 2), int((size_large[1] - im.size[1]) / 2)))
        im2 = background.rotate(angle)
        if background.getbbox():
            imageBox = im2.getbbox()
            im2 = im2.crop(imageBox)
            wr, hr = im2.size
            print(f"Old dim was {w} by {h}, after rotation w = {wr}, h = {hr}")
            if imageBox[2]-imageBox[0] <= max_side_dim and imageBox[3]-imageBox[1] <= max_side_dim:
                background2 = Image.new('RGB', size, (0, 0, 0)) 
                background2.paste(im2, (int((size[0] - im2.size[0]) / 2), int((size[1] - im2.size[1]) / 2)))
                background2.save(outFolder+filename+'_r' + str(angle) + '.jpg', 'JPEG', quality = 95)
                print("Jpeg exported.")
                return 1
            else:
                print("Too large!")
                return 0
        else:
            print("All black!")
            return 2
        
#@profile
def angle_geom(geom,h,w):
    coords_bb = list(box(geom.bounds[0],geom.bounds[1],geom.bounds[2],geom.bounds[3]).exterior.coords)
    h_m = Point(coords_bb[1]).distance(Point(coords_bb[0]))
    w_m = Point(coords_bb[1]).distance(Point(coords_bb[2]))
    print(f"Property {feat['properties']['OBJEKT_ID']} is {w/4:3.2f} by {h/4:3.2f} pixelmeters and {w_m:3.2f} by {h_m:3.2f} coordmeters" )
    for r in range(0,91,1):
        geom_angle = rotate(geom, r, origin='center')
        coords_bb = list(box(geom_angle.bounds[0],geom_angle.bounds[1],geom_angle.bounds[2],geom_angle.bounds[3]).exterior.coords)
        w_angle = Point(coords_bb[1]).distance(Point(coords_bb[0]))
        h_angle = Point(coords_bb[1]).distance(Point(coords_bb[2]))
        if w_angle <= 74.8 and h_angle <= 74.8:
            print(f"Rotation successful, property {feat['properties']['OBJEKT_ID']} is {w_angle} by {h_angle}, angle is {r} degrees" )
            return r
    return 0

inFolder = INPUT_FOLDER
outFolder = OUTPUT_FOLDER
logFolder = LOG_FOLDER
SOURCE_GTIFF = VRT_PATH
SOURCE_SHP = LOT_PATH

min_side_dim = 30
max_side_dim = 300
        
timestr = time.strftime("%Y%m%d-%H%M%S")
logname = LOG_NAME
year = YEAR

gdal.UseExceptions()

with fiona.open(SOURCE_SHP, 'r') as source, open(logFolder+logname,'w', newline='') as out:
    nfeat = len(source)
    csvwriter=csv.writer(out)
    csvwriter.writerow(["OBJEKT_ID","cut","angle","width","height",f"ok_{year}"])
    for idx, feat in enumerate(source):
        print(f"Feat {idx} out of {nfeat}")
        status = feat["properties"][f"ok_{year}"] == 1
        try:
            newfile = feat["properties"]["OBJEKT_ID"]+'_'+str(status)
        except:
            continue
        with fiona.open(path=inFolder+"tempfile.shp", mode='w', driver=source.driver, schema=source.schema, crs=source.crs) as tempshp:
           tempshp.write(feat)
        outtile = gdal.Warp(outFolder+newfile+".tif", SOURCE_GTIFF, format = 'GTiff', cutlineDSName=inFolder+"tempfile.shp", cropToCutline=True)
        outtile = None
        print(newfile+" cutline performed.")
        im = Image.open(outFolder+newfile+".tif", 'r')
        w, h = im.size
        if min(h, w) > min_side_dim and max(h, w) < max_side_dim:
            r = 0
            cut = toJpeg(im,newfile,max_side_dim,outFolder,r)
        elif min(h, w) > min_side_dim and  max(h, w) > max_side_dim:
            geom = shape(feat["geometry"])
            r = angle_geom(geom,h,w)
            print(f"Angle is {r}")
            if r != 0:
                cut = toJpeg(im,newfile,max_side_dim,outFolder,angle=r)
            else:
                print("Too large!")
                cut = 0
        else:
            print("Too small!")
            cut = 0
            r = 0
        csvwriter.writerow([newfile,cut,r,w,h,status])
        try:
            os.remove(outFolder+newfile+".tif")
        except PermissionError:
            continue

for file in os.listdir(os.fsencode(outFolder)):
    filename = os.fsdecode(file)
    if filename.endswith(".tif"):
        try:
            os.remove(outFolder+filename)
        except PermissionError:
            continue
         