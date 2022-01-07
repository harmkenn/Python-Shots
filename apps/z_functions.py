import streamlit as st
import numpy as np
import pandas as pd
import requests
import urllib.request
import json
import pygeodesy as pg 

nd = pd.read_csv('data/northdes.csv')

def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos

# Lat Lon to MGRS
def LL2MGRS(lat,lon):
    # Pick a hemisphere
    hem = 'N'
    if lat<0: hem = 'S'
    # convert to radians
    pi = np.pi
    latr = lat*pi/180
    lonr = lon*pi/180
    
    # Lets go find Easting
    sec1 = pi/(180*3600) #One Second
    a = 6378137 # Equitorial Radius
    b = 6356752.31424518 #Polar Radius
    k0 = .9996 #Scalar Factor Constant
    gzen = np.floor(1/6*lon)+31 #Longitude Zone
    Czone = 6*gzen - 183 #Longitude of the center of the zone
    dlon = lon - Czone # Longitude from the center of the zone
    p = dlon*3600/10000 #Hecta seconds?
    e = np.sqrt(1-(b/a)**2) #eccentricity
    e1 = np.sqrt(a**2-b**2)/b
    e1sq = e1**2
    c = a**2/b
    nu = a/np.sqrt(1-(e*np.sin(latr))**2) #r curv 2
    Kiv = nu*np.cos(latr)*sec1*k0*10000 #Coef for UTM 4
    Kv = (sec1*np.cos(latr))**3*(nu/6)*(1-np.tan(latr)**2+e1sq*np.cos(latr)**2)*k0*10**12 #Coef for UTM 5
    Easting = 500000+Kiv*p+Kv*p**3
    
    # Now let's go find Northing
    n = (a-b)/(a+b)
    A0 = a*(1-n+(5*n**2/4)*(1-n)+(81*n**4/64)*(1-n)) # Meridional Arc Length
    B0 = (3*a*n/2)*(1-n-(7*n**2/8)*(1-n)+55*n**4/64) # Meridional Arc Length
    C0 = (15*a*n**2/16)*(1-n+(3*n**2/4)*(1-n)) # Meridional Arc Length
    D0 = (35*a*n**3/48)*(1-n+11*n**2/16) # Meridional Arc Length
    E0 = (315*a*n**4/51)*(1-n) # Meridional Arc Length
    S = A0*latr - B0*np.sin(2*latr) + C0*np.sin(4*latr) - D0*np.sin(6*latr) + E0*np.sin(8*latr) # Meridional Arc
    Ki = S*k0 #Coef for UTM 1
    Kii = nu*np.sin(latr)*np.cos(latr)*sec1**2*k0*100000000/2 #Coef for UTM 2
    Kiii = ((sec1**4*nu*np.sin(latr)*np.cos(latr)**3)/24)*(5-np.tan(latr)**2+9*e1sq*np.cos(latr)**2*np.cos(latr)**4)*k0*10**16 #Coef for UTM 2
    Northing = Ki + Kii * p**2 + Kiii * p**4
    if lat < 0: Northing = 10000000 + Northing # In the Southern Hemisphere is Northing is measured from the south pole instead of from the equator
  
    Easting = int(np.floor(Easting)) # 6 digit easting
    Northing = int(np.floor(Northing)) # 7 or 8 digit northing
    
    ## Now let's turn UTM into MGRS
    gzoe = 'Odd' 
    if gzen % 2 == 0: gzoe = 'Even' # longitude grid zone
    
    gsen = int(np.floor(Easting/100000)) #Grab off the first digit off the Easting
    gsnn = int(np.floor(Northing/100000)) #Grab off the first two digits off the Northing
    if gzoe =='Even': gsnn = gsnn + 5
    #ck gridsquare letters
    ckgsn = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V'] # northing grid square letter
    gsn = gsnn%20 # Latitude letter zone count
    
    gsnl = ckgsn[gsn] # northing grid zone letter
    
    if gzoe =='Even': gsnn = gsnn - 5
    
    eldf = pd.DataFrame({1:['A','J','S'],2:['B','K','T'],3:['C','L','U'],4:['D','M','V'],5:['E','N','W'],6:['F','P','X'],7:['G','Q','Y'],8:['H','R','Z']},index=[1,2,0])
    gsel = eldf.loc[gzen%3,gsen]
    
    this = pd.DataFrame({'hem':hem,'gzoe':gzoe,'gsnl':gsnl,'gsnn':gsnn},index=[0])
    
    this = this.merge(nd, how='left', on=['hem','gzoe','gsnl','gsnn'])
    
    gznl = this.loc[0,'gznl']
    
    return [str(int(gzen))+str(gznl)+' '+str(Easting).rjust(6, "0")+' '+str(Northing).rjust(8, "0"),
            str(int(gzen))+str(gznl)+str(gsel)+str(gsnl)+str(Easting-gsen*100000).rjust(5, "0")+str(Northing-gsnn*100000).rjust(5, "0")]

# MGRS to Lat Lon
def MGRS2LL(mgrs):
    out = ''.join([i for i in mgrs if not i.isdigit()])
    ppp = mgrs.split(out)
    l = [out[index : index + 1] for index in range(0, 3, 1)]
    ten = [ppp[1][index : index + 5] for index in range(0, 10, 5)]
    gzen = int(ppp[0])
    gznl = l[0]
    gsel = l[1]
    gsnl = l[2]
    east5 = int(ten[0])
    north5 = int(ten[1])
    # grid square easting letters data frame
    gseldf = pd.DataFrame({1:['A','J','S'],2:['B','K','T'],3:['C','L','U'],4:['D','M','V'],5:['E','N','W'],6:['F','P','X'],7:['G','Q','Y'],8:['H','R','Z']},index=[1,2,0])
    
    sec1 = np.pi/(180*3600) #One Second
    a = 6378137 # Equitorial Radius
    b = 6356752.31424518 #Polar Radius
    k0 = .9996 #Scalar Factor Constant
    e1 = np.sqrt(a**2-b**2)/b
    e1sq = e1**2
    c = a**2/b
    
    hem = 'S'
    if gznl in 'NPQRSTUVWX': hem = 'N'
    
    eI = getIndexes(gseldf,gsel)[0][1]
    Easting = eI*100000 + east5
    
    gzoe = 'Odd' 
    if gzen % 2 == 0: gzoe = 'Even'
    this = pd.DataFrame({'hem':hem,'gzoe':gzoe,'gsnl':gsnl,'gznl':gznl},index=[0])
    this = this.merge(nd, how='left', on=['hem','gzoe','gsnl','gznl'])
    gsnn = this.loc[0,'gsnn']
    Northing = gsnn*100000 + north5
    NfEQ = Northing
    if hem == 'S': NfEQ = Northing - 10000000
    Fi = (NfEQ)/(6366197.724*k0)
    Ni = (c/(1+e1sq*(np.cos(Fi))**2)**(1/2))*k0
    Czone = 6*gzen-183
    dln = (Easting-500000)/Ni
    A1 = np.sin(2*Fi)
    A2 = A1*(np.cos(Fi))**2
    J2 = Fi+(A1/2)
    J4 = (3*J2+A2)/4
    J6 = (5*J4+A2*(np.cos(Fi))**2)/3
    alfa = 3/4*e1sq
    beta = 5/3*alfa**2
    gamma = 35/27*alfa**3
    Bfi = k0*c*(Fi-(alfa*J2)+(beta*J4)-(gamma*J6))
    BB = (NfEQ-Bfi)/Ni
    zeta = ((e1sq*dln**2)/2)*(np.cos(Fi))**2
    Xi = dln*(1-(zeta/3))
    Eta = BB*(1-zeta)+Fi
    ShXi = (np.exp(Xi)-np.exp(-Xi))/2
    dLam = np.arctan(ShXi/np.cos(Eta))
    Tau = np.arctan(np.cos(dLam)*np.tan(Eta))
    FiR = Fi+(1+e1sq*(np.cos(Fi))**2-(3/2)*e1sq*np.sin(Fi)*np.cos(Fi)*(Tau-Fi))*(Tau-Fi)
    lat = FiR/np.pi*180
    lon = dLam/np.pi*180+Czone
    
    return [str(gzen)+str(gznl)+' '+str(Easting).rjust(6, "0")+' '+str(Northing).rjust(8, "0"),lat,lon]
    
def lookup(what):
    response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address='+ what +'&key=AIzaSyAfnLNZjvYdMx-cyga_qA1oJ6P36dRGalA')
    resp_json_payload = response.json()
    address = resp_json_payload['results'][0]['formatted_address']
    lat = resp_json_payload['results'][0]['geometry']['location']['lat']
    lon = resp_json_payload['results'][0]['geometry']['location']['lng']
    mgrs = LL2MGRS(lat,lon)
    return [address,lat,lon,mgrs[1]]

def elevation(lat, lng):
    with urllib.request.urlopen('https://maps.googleapis.com/maps/api/elevation/json?'+"locations=%s,%s&sensor=%s" % (lat, lng, "false")+'&key=AIzaSyAfnLNZjvYdMx-cyga_qA1oJ6P36dRGalA') as f:
        response = json.loads(f.read().decode())    
        return response['results'][0]['elevation']

def polar2LL(lat,lon,dir,dist):
    # to radians
    latr = lat*np.pi/180
    lonr = lon*np.pi/180
    dirr = dir*np.pi/180
    er = 6371 #Earth Radius in km
    delta = dist/er
    
    later = np.arcsin(np.sin(latr)*np.cos(delta)+np.cos(latr)*np.sin(delta)*np.cos(dirr))
    loner = lonr + np.arctan2(np.sin(dirr)*np.sin(delta)*np.cos(latr),np.cos(delta)-np.sin(latr)*np.sin(later))
    
    # Back to Degrees
    lated = later*180/np.pi
    loned = loner*180/np.pi
    
    # Impact Bearing in Radians
    impr = np.pi + np.arctan2(np.sin(lonr-loner)*np.cos(latr),np.cos(later)*np.sin(latr)-np.sin(later)*np.cos(latr)*np.cos(lonr-loner))
    
    # Impact Bearing in Degrees
    impd = impr*180/np.pi
    
    # compute the midpoint in Radians
    Bx = np.cos(later)*np.cos(loner-lonr)
    By = np.cos(later)*np.sin(loner-lonr)
    latmr = np.arctan2(np.sin(latr) + np.sin(later), np.sqrt((np.cos(latr)+Bx)**2+By**2))
    lonmr = lonr + np.arctan2(By, np.cos(latr) + Bx)
    # midpoint in degrees
    latmd = latmr*180/np.pi
    lonmd = lonmr*180/np.pi
    
    return [lated, loned, impd,latmd,lonmd]

def P2P(lat1d,lon1d,lat2d,lon2d):
    lbd = pg.bearing(lat1d,lon1d,lat2d,lon2d) #Launch Bearing in degrees
    ibd = pg.bearing(lat2d,lon2d,lat1d,lon1d)-180 #impact Bearing in degrees
    if ibd < 0: ibd = ibd + 360

    # six different distances
    dist = pg.cosineForsytheAndoyerLambert(lat1d,lon1d,lat2d,lon2d) # This is the best one   
    
    return [lbd,ibd,dist]