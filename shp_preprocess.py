# -*- coding: iso-8859-1 -*-
"""
Created on Thu Apr 19 16:10:32 2018

@author: kalle
"""

import fiona
from shapely.geometry import shape, mapping
from rtree import index

#match properties to buildings
inFolder = INPUT_FOLDER
outFolder = OUTPUT_FOLDER
        
BUILDING_SHP = inFolder+'by_all.shp'
LOT_SHP = inFolder+'ay_all.shp'
TARGET_SHP = outFolder+'Lots_all_10000.shp'
POINTS_SHP = outFolder+"BuildingCentroids.shp"
    
maxArea = 10000
             
source = fiona.open(LOT_SHP, 'r', encoding='iso-8859-1')              
    
pointSchema =  {'geometry': 'Point',
                   'properties': {'OBJEKT_ID': 'str'}}

with fiona.open(POINTS_SHP, 'w', driver=source.driver, crs=source.crs, schema=pointSchema) as dest, fiona.open(BUILDING_SHP, encoding='iso-8859-1') as buildings:
    for feat in buildings:
        typecode = feat['properties']['ANDAMAL_1']
        #select typecodes corresponding to singe-family homes
        if ((typecode == 130) or (typecode == 131) or (typecode == 132)):
            geom = shape(feat["geometry"])
            dest.write({'geometry': mapping(geom.centroid), 'properties':{'OBJEKT_ID' : feat['properties']['OBJEKT_ID']}})
            print(feat['properties']['OBJEKT_ID'])

# Create the R-tree index and store the features in it (bounding box)                
polygons = [pol for pol in source]
points = [pt for pt in fiona.open(POINTS_SHP)]
    
idx = index.Index()
for pos, poly in enumerate(polygons):
    idx.insert(pos, shape(poly['geometry']).bounds)
    
    #iterate through points
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