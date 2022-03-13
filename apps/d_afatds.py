import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import plugins
import numpy as np
from sklearn.linear_model import ElasticNet

from apps import z_functions as zf

def app():
    # title of the app
    st.markdown('AFATDS')
    c1,c2,c3 = st.columns((1,3,1.5))
    with c1:
        lu = st.sidebar.text_input('Lookup: ') 
        if len(lu)>=3:
            where = zf.lookup(lu)
            st.sidebar.write(where[0])
            st.sidebar.write('Lat: '+str(where[1])+' Lon: '+str(where[2]))
            st.sidebar.write('MGRS: '+where[3])
            alt = zf.elevation(where[1],where[2])
            st.sidebar.write('Alt :'+str(round(alt,2))+' Meters')
    with c1:
        lpmgrs = st.text_input('Launch Point (MGRS):','12RWU1059022575')
        lpalt = st.text_input('Launch Altitude (M)', 321)
        ipmgrs = st.text_input('Impact Point (MGRS):','12RWU2645019206')
        ipalt = st.text_input('Impact Altitude (M)', 621)
        AOF = st.text_input('Azimuth of Fire', 1200)
        
    if len(lpmgrs)>3 and len(ipmgrs)>3:
        lp = zf.MGRS2LL(lpmgrs)
        ip = zf.MGRS2LL(ipmgrs)
        with c1:
            deets = zf.P2P(lp[1],lp[2],ip[1],ip[2])
            st.write('Distance: ' + str(round(deets[2],0)) + ' meters')
            st.write('Launch Bearing: '+str(round(deets[0],2)) + ' degrees')
            st.write('Launch Azimuth: '+str(round(deets[0]*3200/180,2)) + ' mils')
            st.write('Impact Bearing: '+str(round(deets[1],2)) + ' degrees')
            st.write('Impact Azimuth: '+str(round(deets[1]*3200/180,2)) + ' mils')
        with c2:
            # map
            map = folium.Map(location=[(lp[1]+ip[1])/2, (lp[2]+ip[2])/2], zoom_start=-1.36*np.log(deets[2]/1000)+15)
            # add tiles to map
            folium.raster_layers.TileLayer('Open Street Map').add_to(map)
            folium.raster_layers.TileLayer('Stamen Terrain').add_to(map)
            # Add custom base maps to folium
            folium.raster_layers.TileLayer(
                    tiles = 'https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
                    attr = 'Google',
                    name = 'Google Maps',
                    overlay = False,
                    control = True
                ).add_to(map)
            folium.raster_layers.TileLayer(
                    tiles = 'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                    attr = 'Google',
                    name = 'Google Satellite',
                    overlay = False,
                    control = True
                ).add_to(map)
            folium.raster_layers.TileLayer(
                    iles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
                    attr = 'Google',
                    name = 'Google Terrain',
                    overlay = False,
                    control = True
                ).add_to(map)

            # add layer control to show different maps
            folium.LayerControl().add_to(map)
            
            # plugin for mini map
            minimap = plugins.MiniMap(toggle_display=True)

            # add minimap to map
            map.add_child(minimap)
            
            # add scroll zoom toggler to map
            plugins.ScrollZoomToggler().add_to(map)

            # add full screen button to map
            plugins.Fullscreen(position='topright').add_to(map)
            
            # add marker to map https://fontawesome.com/v5.15/icons?d=gallery&p=2&m=free
            pal = folium.features.CustomIcon('Icons/paladin.jpg',icon_size=(30,20))
            tgt = folium.features.CustomIcon('Icons/target.png',icon_size=(25,25))
            folium.Marker(location=[lp[1],lp[2]], color='green',popup=lpmgrs, tooltip='Launch Point',icon=pal).add_to(map)
            folium.Marker(location=[ip[1],ip[2]], color='green',popup=ipmgrs, tooltip='Impact Point',icon=tgt).add_to(map)
            

            
        with c2:
            points = []
            points.append([lp[1],lp[2]])
            for p in range(1,101):
                get = zf.polar2LL(lp[1],lp[2],deets[0],deets[2]*p/100000)
                points.append([get[0],get[1]])
            points.append([ip[1],ip[2]])
            folium.PolyLine(points, color='red').add_to(map)
            
            
        
            
            draw = plugins.Draw()
            draw.add_to(map)
            # display map
            folium_static(map) 
        with c3:
            rng = round(deets[2],0)
            chrg = st.selectbox('Charge:',['Auto','M231 1L','M231 2L','M232A1 3H','M232A1 4H','M232A1 5H'])
            if chrg == 'Auto':
                chrg = pd.cut([rng], bins=[-1,2000,5000,8000,11000,14000,25000,99999999], labels=['Too Short','M231 1L','M231 2L','M232A1 3H','M232A1 4H','M232A1 5H','Too Far'])
                chrg = chrg[0]
            st.write(chrg)
            ning = int(lpmgrs[-5:])
            if ning > 1000:
                ring = ning - 1000
                sp = lpmgrs[:-5]+str(ring).rjust(5, "0")
            else:
                ring = ning + 1000
                sp = lpmgrs[:-5]+str(ring).rjust(5, "0")
            spll = zf.MGRS2LL(sp)
            gd = zf.P2P(spll[1],spll[2],lp[1],lp[2])
            gdm = gd[0]/180*3200
            if gdm > 4800:
                gdm = gdm - 6400
            if gdm > 1600:
                gdm = gdm - 3200
            macs = pd.read_csv('data/MACs.csv',encoding = 'latin1')
            
            macs = macs[macs['Charge'].str.contains(chrg[-2:])]
            st.write(macs)
            DriftM = ElasticNet()
            DriftM.fit(macs[['Range']],macs['Drift'])
            drift = DriftM.predict([[rng]])[0]
            az = deets[0]*3200/180
            defl = 3200 - az + float(AOF) + drift + gdm
            mv = macs.iloc[1,1]
            ElevM = ElasticNet()
            ElevM.fit(macs[['Range']],macs['Elev'])
            elev = ElevM.predict([[rng]])[0]
            data = pd.DataFrame({'Range (Meters)':str(rng),'Shell':'M795','Charge':chrg,'Azimuth to Target (mils)':str(round(az,2)),
                                 'Grid Declination (mils)':str(round(gdm,1)),'Drift':str(round(drift,1)),'Deflection (mils)':str(round(defl,1)),
                                 'Muzzle Velocity (m/s)':str(mv),'Elevation (mils)':str(round(elev,1))},index = ['Fire Mission']).T 
            
            st.write(data) 
            
            


        
    