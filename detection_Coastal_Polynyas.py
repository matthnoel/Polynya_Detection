#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:50:58 2023

@author: mnoel
"""

# In[Chargement bibliothèques]
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
from scipy import stats
import datetime
from math import radians,degrees
import math as m

# Loading of the flood_fill function
from skimage.segmentation import flood_fill

# In[creation DataArray]

# =============================================================================
# FONCTION : creationDataArray :
#
#   Creation of a dataArray from existing array.
#
#  Input :
#    - data (array - with 2 or 3 dimension) : data to work on 
#    - Latitudes (list or array (1 d))) : list of latitudes (size of the fisrt dimension of data).
#    - Longitudes (list or array (1 d))) : list of longitudes (size of the second dimension of data).
#    - time (list or array (1 d))) : list of time (size of the last dimension of data) - None by default.
#
#  Output :
#     - dataArray (xarray) : xarray created from data (Latitudes, Longitudes,time (or not)).
#   
# =============================================================================

def creationDataArray(Data,Latitudes,Longitudes,time = None,nameDataArray="DataArray"):
    #print('time creation DataArray : ',time)
    # Creation d'un xarray de données issue de l'algo
    shape = Data.shape
    n = len(shape)
    
    # Test of the dimension of date (Is there a dimension for time ?)
    if (n == 2):
        # print('time is none')
        
        # Creation of the dataArray from Latitudes and Longitudes.
        data_DataArray = xr.DataArray(
            
            ## Valeurs du tableau de données        
            data = Data
            ,
            coords=dict(
                ## Coordonnées Latitudes
                latitude=(["latitude"], Latitudes)
            ,
                ## Coordonnées Longitudes
                longitude=(["longitude"], Longitudes)
            ),
            attrs=dict(
                description = nameDataArray
                )
        )
    elif (n == 3):
        # Verification if there a parameter for time (else creation by default)
        if (time is None):
            time = range(shape[0])
        
        # Creation of the dataArray
        data_DataArray = xr.DataArray(
            
            ## Valeurs du tableau de données        
            data = Data
            ,
            coords=dict(
                time=(["time"], time)
            ,
                ## Coordonnées Latitudes
                latitude=(["latitude"], Latitudes)
            ,
                ## Coordonnées Longitudes
                longitude=(["longitude"], Longitudes)
            ),
            attrs=dict(
                description = nameDataArray
                )
        )
    # data is not 2d or 3d --> Error and output is None
    else :
        print('Error : data is not 2 or 3 dimension')
        data_DataArray = None
    
    return data_DataArray

# In[Algotihm of floodFill applied for the polynyas]

# =============================================================================
# FONCTION : floodFillPolynya :
#
#   Calculation of the surface of the polynyas
#
#  Input :
#    - data_ci (xarray - 2 dimensions) - data_ci(Latitudes,Longitudes) : sea ice couverture around Antarctica for 1 day.
#    - lon_dep (float) : longitude of the point of beginning of injection.
#    - lat_dep (float) : latitude of the point of beginning of injection
#    - continent (bool) : parameter to describe if you are on the continent or not (True by default).
#    - new_value (float) : new value of sea ice couverture on each point.
#    - tolerance (float) : gap of reference for the difference of value between the beginning point and the other one.
#    - trace (bool) : parameter to describe if you draw maps or not during the process.
#    - reanalysis (str) : type of reanalysis for the data.
#
#  Output :
#     - data_filled_ci (xarray) - data_filled_ci(Latitudes,Longitudes) : xarray of sea ice couverture after using the flood fill algorithm
#   
# =============================================================================

def floodFillPolynya(data_ci,lon_dep,lat_dep,ocean = True,new_value= 100,tolerance = 30 ,trace = False,CIlevels = None,CIlevels2 = None,reanalysis='ERA'):
    
    # Multiplication of the parameter ci by 100 to work with int (%)
    ci = data_ci*100
    
    # Loading the useful data 
    if reanalysis == 'ERA':
        Lat = ci.latitude.values
        Lon = ci.longitude.values
        
    if reanalysis == 'MERRA':
        Lat = ci.lat.values
        Lon = ci.lon.values
    
    # Creation of the output array
    ci_int = np.zeros(ci.shape)
    
    # Test if you are on the continent or not (NB : in ERA5, on the continent ci = Nan).
    for i in range(len(ci_int)):
        for j in range(len(ci_int[0])):
            valeur = np.array(ci)[i,j]
            
            if (not np.isnan(valeur)):
                ci_int[i,j] = int(valeur)
            else :
                if ocean :
                    ci_int[i,j] = valeur
                else :
                    ci_int[i,j] = 0
    
    # Dtermination of the coordinates of the starting point
    i_lon = 0
    i_lat = 0
    for i in range(len(Lat)):
        if (Lat[i] == lat_dep):
            i_lat = i
    for i in range(len(Lon)):
        if (Lon[i] == lon_dep):
            i_lon = i
    
    # We apply the daughter flood algorithm with the following starting point (latitude,longitude)
    # we set a new value and a tolerance
    filled_ci = flood_fill(ci_int,(i_lat,i_lon),new_value, tolerance = tolerance)*0.01

    # Creation of a xarray of the data
    data_filled_ci = creationDataArray(filled_ci,Lat,Lon,nameDataArray="Filled ci.")

    # if trace :
    #     plotMap_Ant(data_filled_ci,CIlevels)
    
    return data_filled_ci 
    

# In[Detections Polynies - Antarctique]

# =============================================================================
# FONCTION : detectionPolynya :
#
#   Detection of the polynyas in Antractica from a map of sea ice couverture (ci)
#
#  Input :
#    - ci_antarctique (xarray - 2 dimensions) - ci_antarctique(Latitudes,Longitudes) : sea ice couverture around Antarctica for 1 day.
#    - ci_limit (float in [0,1]) : limit concentration of sea ice below which we consider there is polynyas
#    - trace (bool) : boolean allowing to draw the maps of the algorithm progress
#    - reanalysis (str) : precision on the type of reanalysis used (difference of the name of parameters).
#
#  Output :
#     - final_polynyas (xarray) - final_polynyas(Latitudes,Longitudes) : xarray of polynya index (1 if you are in a polynya, 0 else)
#   
# =============================================================================

def detectionPolynya(ci_antarctique, ci_limit = 0.7, trace=False, reanalysis='ERA'):
    Levels = np.linspace(0,100,21)
    CIlevels = np.linspace(0,1,11)
    # CIlevels2 = np.linspace(0,1,21)
    
    # Loading of the 
    if reanalysis == 'ERA':
        
        Latitudes = ci_antarctique.latitude.values
        Longitudes = ci_antarctique.longitude.values
        
    if reanalysis == 'MERRA':
        
        Latitudes = ci_antarctique.lat.values
        Longitudes = ci_antarctique.lon.values
    
    # Draw the originla map of the sea ice cover
    # if trace :
    #     plotMap_Ant(ci_antarctique,dimension="%",color='jet',titre='Carte entrée')#,titre='Map CI Antarctica - couverture normale')

    # Application of the floodFillPolynya to the original map of sea ice couverture, 
    # from the starting point : (longitude,latitude) = (240,-60) (in a n area of open water even if in winter)
    # We fix the tolerance at 100*ci_limit
    filled_ci_2014_Sep_m1 = floodFillPolynya(ci_antarctique,
                                             240,
                                             -60,
                                             trace=False,
                                             tolerance=ci_limit*100,
                                             CIlevels= Levels,
                                             reanalysis=reanalysis)

    # Draw the originla map of the sea ice cover
    # if trace :
    #     plotMap_Ant(filled_ci_2014_Sep_m1, dimension="%",color='jet')#,titre='Map CI Antarctica - couverture glace en mer')
    
    # Application of the filter (depends on ci_limit) to determine the polynyas (coastal and open water)
    Polinies = np.zeros(filled_ci_2014_Sep_m1.values.shape)
    for i in range(Polinies.shape[0]):
        for j in range(Polinies.shape[1]):
            
            # if the sea ice couverture is less than the limit --> It is a polynya
            if ((filled_ci_2014_Sep_m1.values[i,j] <ci_limit) and (filled_ci_2014_Sep_m1.values[i,j] >= 0)):
                Polinies[i,j] = 1
    
    # Creation of a dataArray of the polynyas detected (coastal and OWP)
    data_polinies = creationDataArray(Polinies, Latitudes, Longitudes, nameDataArray="Filled ci.")
    
    # if trace :
    #     plotMap_Ant(data_polinies,CIlevels,color='binary',color_bar=False)#,titre='Map CI Antarctica - Polynies et OWC')
    
    
    # Application of the floodFillPolynya to the original map of sea ice couverture, 
    # from the starting point : (longitude,latitude) = (90,-89) (on the continent)
    # Here, we fill the continent and the coastl polynya with sea ice
    filled_ci_2014_Sep_m2 = floodFillPolynya(filled_ci_2014_Sep_m1,
                                             90,
                                             -89,
                                             new_value=100,
                                             ocean = False,
                                             trace=False,
                                             tolerance=ci_limit*100,
                                             CIlevels2= CIlevels)
    
    # Application of the filter (depends on ci_limit) to determine the Open Water Polynyas
    OWP = np.zeros(filled_ci_2014_Sep_m1.values.shape)
    for i in range(OWP.shape[0]):
        for j in range(OWP.shape[1]):
            if ((filled_ci_2014_Sep_m2.values[i,j] < ci_limit) and (filled_ci_2014_Sep_m2.values[i,j] >= 0)):
                OWP[i,j] = 1
    
    # Creation of a dataArray of the polynyas detected (only OWP)
    data_OWP = creationDataArray(OWP, Latitudes, Longitudes,nameDataArray="Filled ci.")
    
    # Difference of the data of all polynya and of the Open Water Polynya to determine the coastal polynyas only
    final_polynyas = data_polinies - data_OWP
    
    # if trace :
    #     plotMap_Ant(final_polynyas,CIlevels,color='binary',color_bar=False)#,titre='Map CI Antarctica - polynies seules')

    # print('Surface des polynies = ',calculSurfacePolynies(sel_CapeDarnley(final_polynyas),area='CD'), ' km2')
    
    # Return the index of costal polynyas --> give the positions of polynyas around the Antarctica
    return final_polynyas