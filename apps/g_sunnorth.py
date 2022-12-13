import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import folium
from streamlit_folium import folium_static
from folium import plugins
from apps import z_functions as zf
def app():
    # title of the app
    st.markdown('Get your position from the location of the sun')
    c1,c2 = st.columns((1,3))
    with c1:
        
        setyear = st.number_input('Year: ',1900,2100,2022)
        setmonth = st.number_input('Month: ',1,12,4)
        setday = st.number_input('Day: ',1,31,9)
        sethour = st.number_input('Hour: ',0,23,22)
        setmin = st.number_input('Minute: ',0,59,40)
        setsec = st.number_input('Second: ',0,59,0)
        
        setday = datetime(setyear,setmonth,setday,sethour,setmin,setsec)

        

    with c2:
        if st.button('Now'):
            setday = datetime.now(timezone.utc)
        st.write(str(setday)+'UTC')
        sslon = 180-(setday.hour+setday.minute/60+setday.second/3600)*15+22/60
        
        julian = int(setday.strftime('%j')) + (setday.hour+setday.minute/60+setday.second/3600)/24
        a = 360/365.24*(julian - 2)*np.pi/180
        b = (360/365.24*(julian+10)+360/np.pi*.0167*np.sin(a))*np.pi/180
        c = np.sin(-23.44*np.pi/180)*np.cos(b)
        sslat = np.arcsin(c)*180/np.pi - .07
        st.write('Sub Solar Point: '+str(sslat)+', '+str(sslon),', MGRS: '+ zf.LL2MGRS(sslat,sslon)[1])
        lpmgrs = st.text_input('Your Location (MGRS):',zf.LL2MGRS(sslat, sslon)[1])

       
       
    
        
        # map
        map = folium.Map(location=[0, 0], zoom_start=1)
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
        sun = folium.features.CustomIcon('Icons/target.png',icon_size=(30,30))
        folium.Marker(location=[sslat, sslon], color='green', tooltip='SubSolar Point',icon=sun).add_to(map)
        pal = folium.features.CustomIcon('Icons/paladin.jpg',icon_size=(30,20))
        folium.Marker(location=[melat, melon], color='green', tooltip='my location',icon=pal).add_to(map)
        folium.PolyLine([[sslat, sslon],[melat,melon]],tooltip='Azimuth').add_to(map)
        draw = plugins.Draw()
        draw.add_to(map)
        # display map
        folium_static(map)

        
        

    