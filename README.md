# Trampoline detection using CNN

## Intro

This repo contains some code used in my research project on neighborhood effects in consumption. You can download a working paper on my personal [website](http://erikgrenestam.se/wp-content/uploads/2019/04/Bouncing-with-the-Joneses-ErikG.pdf).

Trampolines are popular among Swedish families. Due to their size and distinct shape, they can be detected from an aerial photo. To collect data on trampoline ownership, I apply an instance of Inception ResNet to a large set of aerial photos of Swedish neighborhoods taken between 2006 and 2018.

## Image preprocessing

Each raw image is a 10,000 by 10,000 pixel GeoTIFF. As my data contains several thousand images, importing all of them into QGIS is not feasible. In ```geotiff_coverage.py```, I extract metadata from each image and it's corresponding json meta file and collect the geographic extent of each image in a shapefile:

```python
import fiona
from shapely.geometry import mapping, shape, box, Point
import os, os.path
from osgeo import gdal

data = gdal.Open(picFolder+file+'.tif', gdal.GA_ReadOnly)
geoTransform = data.GetGeoTransform()
minx = geoTransform[0]
maxy = geoTransform[3]
maxx = minx + geoTransform[1] * data.RasterXSize
miny = maxy + geoTransform[5] * data.RasterYSize
geom = box(minx,miny,maxx,maxy)
img_id = file[0:11]
img_year = file[12:]
yeard[picFolder+file+'.tif'] = int(img_year)
metapath = metaFolder+img_id+'_flygbild_'+img_year+'.json'

with open(metapath) as metafile:    
    metadata = json.load(metafile)
    try:
        img_date = metadata['features'][0]['properties']['tidpunkt'][0:10]
        img_camera = metadata['features'][0]['properties']['kamera']
        dest.write({'geometry': mapping(geom), 'properties':{ 'FILENAME' : file, 
                                                              'ID' : img_id, 
                                                              'DATE': img_date,
                                                              'YEAR': img_year,
                                                              'CAMERA': img_camera}})

```

Using GDAL, I then create virtual mosaics (VRT) containing all the photos from a particular year. To slice each image into chips, each containing a single-family propoerty, I use administrative shapefiles containing the centroid coordinates of every building and polygons for each property parcel. From ```shp_preprocess.py```:

```python
BUILDING_SHP = 'by_all.shp'
LOT_SHP = 'ay_all.shp'
```

Using the building code field in  ```by_all.shp```, I select the subset of land parcels that intersect with a single-family home:

```python
import fiona
from shapely.geometry import shape, mapping

TARGET_SHP = outFolder+'Lots_all_10000.shp'

maxArea = 10000
             
source = fiona.open(LOT_SHP, 'r', encoding='iso-8859-1')     

polygons = [pol for pol in source]
points = [pt for pt in fiona.open(POINTS_SHP)]
    
idx = index.Index()
for pos, poly in enumerate(polygons):
    idx.insert(pos, shape(poly['geometry']).bounds)
    
    #iterate through points
    with fiona.open(TARGET_SHP, 'w', driver=source.driver, crs=source.crs, schema=source.schema) as dest:
        for i,pt in enumerate(points):
            point = shape(pt['geometry'])
            minArea = maxArea
            minpoly = {}
            # iterate through spatial index
            for j in idx.intersection(point.coords[0]):
                print(point.coords)
                geom = shape(polygons[j]["geometry"])
                area = geom.area
                if point.within(geom) and area < maxArea:
                    #only write smallest matching property for each point
                    if area < minArea:
                        minArea = area
                        minpoly = polygons[j]
                if len(minpoly) > 0:
                    print(pt['id'])
                    dest.write(minpoly)
```

With the relevant properties selected, ```cut_imgs.py``` slices the images to 300 by 300 jpegs (about 75 by 75 meters on the ground). By iterating over the featues in ```TARGET_SHP``` created about, I apply ```gdalwarp``` to the photos with each property outlining the cut. I also set a minimum dimension to discard lots that are too small (30 pixels corresponds to about 7.5 meters):

```python
import fiona
from shapely.geometry import shape, box, Point
from shapely.affinity import rotate
import os, os.path
from PIL import Image
from osgeo import gdal

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

min_side_dim = 30
max_side_dim = 300

for idx, feat in enumerate(source):
        with fiona.open(path=inFolder+"tempfile.shp", mode='w', driver=source.driver, schema=source.schema, crs=source.crs) as tempshp:
            tempshp.write(feat)
        outtile = gdal.Warp(outFolder+newfile+".tif", SOURCE_GTIFF, format = 'GTiff', cutlineDSName=inFolder+"tempfile.shp",                                         cropToCutline=True)
        outtile = None
        im = Image.open(outFolder+newfile+".tif", 'r')
        w, h = im.size
        if min(h, w) > min_side_dim and max(h, w) < max_side_dim:
            r = 0
            cut = toJpeg(im,newfile,max_side_dim,outFolder,r)
```
