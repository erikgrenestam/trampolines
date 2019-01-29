# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:42:09 2018

@author: Erik
"""

import fiona
from shapely.geometry import mapping, shape, box, Point
import os, os.path
from osgeo import gdal
import glob
import json

picFolders = [IMAGE_FOLDERS]
metaFolders = [METADATA_FOLDERS]
outFolder = OUTPUT_FOLDER

yeard = {}

covSchema =  {'geometry': 'Polygon', 'properties': {'FILENAME': 'str',
                                                    'ID': 'str',
                                                    'DATE': 'str',
                                                    'YEAR': 'int',
                                                    'CAMERA': 'str',}}

with fiona.open(outFolder+'tiffCoverage.shp', 'w', driver='ESRI Shapefile', crs='epsg:3006', schema=covSchema) as dest:
    for picFolder, metaFolder in zip(picFolders,metaFolders):
        files = [os.path.splitext(os.path.basename(x))[0] for x in glob.glob(picFolder + '/*.tif')]
        for file in files:
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
                except:
                    dest.write({'geometry': mapping(geom), 'properties':{ 'FILENAME' : file, 
                                                                                'ID' : img_id, 
                                                                                'DATE': 'NA',
                                                                                'YEAR': img_year,
                                                                                'CAMERA': 'NA'}})
                    
            else:
                print(f"No metadata found for {img_id}!")
                break
            
#write lists for vrt
with open('D:/Studsmatta/GIS/vrt/2006.txt', 'w') as f2006, \
     open('D:/Studsmatta/GIS/vrt/2007.txt', 'w') as f2007, \
     open('D:/Studsmatta/GIS/vrt/2008.txt', 'w') as f2008, \
     open('D:/Studsmatta/GIS/vrt/2009.txt', 'w') as f2009, \
     open('D:/Studsmatta/GIS/vrt/2010.txt', 'w') as f2010, \
     open('D:/Studsmatta/GIS/vrt/2011.txt', 'w') as f2011, \
     open('D:/Studsmatta/GIS/vrt/2012.txt', 'w') as f2012, \
     open('D:/Studsmatta/GIS/vrt/2013.txt', 'w') as f2013, \
     open('D:/Studsmatta/GIS/vrt/2014.txt', 'w') as f2014, \
     open('D:/Studsmatta/GIS/vrt/2015.txt', 'w') as f2015, \
     open('D:/Studsmatta/GIS/vrt/2016.txt', 'w') as f2016, \
     open('D:/Studsmatta/GIS/vrt/2017.txt', 'w') as f2017:
         for key, value in yeard.items():
            if value == 2006:
                 f2006.write(key+'\n')
            elif value == 2007:
                f2007.write(key+'\n')
            elif value == 2008:
                f2008.write(key+'\n')
            elif value == 2009:
                 f2009.write(key+'\n')
            elif value == 2010:
                f2010.write(key+'\n')
            elif value == 2011:
                f2011.write(key+'\n')
            elif value == 2012:
                f2012.write(key+'\n')
            elif value == 2013:
                f2013.write(key+'\n')
            elif value == 2014:
                f2014.write(key+'\n')
            elif value == 2015:
                f2015.write(key+'\n')      
            elif value == 2016:
                f2016.write(key+'\n')
            elif value == 2017:
                f2017.write(key+'\n')
"""       
polygons = [pol for pol in LOT_SHP]
tiffs = [sq for sq in fiona.open(inFolder+'tiffCoverage.shp')]    

idx = index.Index()
for pos, poly in enumerate(polygons):
    idx.insert(pos, shape(poly['geometry']).bounds)
    
        
with fiona.open(inFolder+'tiffCoverage.shp', 'r') as source:
    for feat in source:
    newfile = feat["properties"]["FILENAME"]
        with fiona.open(inFolder+newfile+'.shp', 'w') as dest
        
        with fiona.open(TARGET_SHP, 'w', driver=source.driver, crs=source.crs, schema=source.schema) as dest:
    #print(dest.schema)
    for i,pt in enumerate(points):
        point = shape(pt['geometry'])
        # iterate through spatial index
        minArea = maxArea
        minpoly = {}
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
"""