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
metapath_orto = metaFolder+img_id+'_ortofoto_'+img_year+'.json'

if os.path.isfile(metapath) and os.path.isfile(metapath_orto):
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
