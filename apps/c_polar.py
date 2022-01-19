import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import plugins
import numpy as np

from apps import z_functions as zf

def app():
    # title of the app
    st.markdown('Polar Shot')
    c1,c2 = st.columns((1,3))
    with c1:
        lu = st.text_input('Lookup: ') 
        if len(lu)>=3:
            where = zf.lookup(lu)
            st.write(where[0])
            st.write('Lat: '+str(where[1])+' Lon: '+str(where[2]))
            st.write('MGRS: '+where[3])
            alt = zf.elevation(where[1],where[2])
            st.write('Alt :'+str(round(alt,2))+' Meters')
    with c1:
        
        lpmgrs = st.text_input('Launch Point (MGRS):','12RWU1059022575')
        azmils = st.text_input('Azimuth (mils):',1600)
        dmeters = st.text_input('Direction (meters)',10000)
    
    if len(lpmgrs)>3 and len(azmils)>1:
        lp = zf.MGRS2LL(lpmgrs)
        azdeg = float(azmils)*180/3200
        dmeters = float(dmeters)
        with c1:
            deets = zf.polar2LL(float(lp[1]),float(lp[2]),azdeg,dmeters/1000)
            st.write('Distance: ' + str(dmeters) + ' meters')
            st.write('Launch Bearing: '+str(round(azdeg,2)) + ' degrees')
            st.write('Launch Azimuth: '+str(round(float(azmils),2)) + ' mils')
            st.write('Impact Bearing: '+str(round(deets[2],2)) + ' degrees')
            st.write('Impact Azimuth: '+str(round(deets[2]*3200/180,2)) + ' mils')
            ip = zf.LL2MGRS(deets[0],deets[1])
            st.write('Impact Location (MGRS): '+ip[1])
        with c2:
            # map
            map = folium.Map(location=[(lp[1]+deets[0])/2, (lp[2]+deets[1])/2], zoom_start=-1.36*np.log(dmeters/1000)+15)
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
            folium.Marker(location=[deets[0],deets[1]], color='green',popup=ip[1], tooltip='Impact Point',icon=tgt).add_to(map)
            

            
        with c2:
            points = []
            points.append([lp[1],lp[2]])
            for p in range(1,101):
                get = zf.polar2LL(lp[1],lp[2],azdeg,p*dmeters/100000)
                points.append([get[0],get[1]])
            points.append([deets[0],deets[1]])
            folium.PolyLine(points, color='red').add_to(map)
            
            
        
            
            draw = plugins.Draw()
            draw.add_to(map)
            # display map
            folium_static(map) 
            
            


        
    