import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import plugins

from apps import z_functions as zf

def app():
    # title of the app
    st.markdown('Deflection')
    c1,c2 = st.columns(2)
    with c1:
        lat = float(st.text_input('Latitude',-30))
        lon = float(st.text_input('Longitude',30))
        back = zf.LL2MGRS(lat,lon)
        st.write('UTM: ',back[0])
        st.write('MGRS: ',back[1])
    with c2:    
        mgrs = st.text_input('MGRS: ',back[1])
        out = zf.MGRS2LL(mgrs)
        st.write('UTM :',out[0])
        st.write('Lat: ',str(round(out[1],4)),' Lon: ',str(round(out[2],4)))
        lu = st.text_input('Lookup: ')
        if len(lu)>=3:
            where = zf.lookup(lu)
            st.write(where[0])
            st.write('Lat: '+str(where[1])+' Lon: '+str(where[2]))
            st.write('MGRS: '+where[3])
            alt = zf.elevation(where[1],where[2])
            st.write('Alt :'+str(round(alt,2))+' Meters')
    d1,d2 = st.columns((1,2))
    with d1:
        aof = st.text_input('Target; Azimuth of Fire (mils): ',2000)
        lpmgrs = st.text_input('Launch Point (MGRS):',back[1])
        ipmgrs = st.text_input('Impact Point (MGRS):')
        lp = zf.MGRS2LL(lpmgrs)
        
        ip = zf.MGRS2LL(ipmgrs)
    with d2:
        # map
        map = folium.Map(location=[lp[1], lp[2]], zoom_start=10)
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
        # radius of the circle in meters
        folium.Circle(radius=30000, location=[lp[1],lp[2]], color='green').add_to(map)
        # add Azimuth of Fire to map
        #aoflat
        #folium.PolyLine().add_to(map)
        
        draw = plugins.Draw()
        draw.add_to(map)
        # display map
        folium_static(map) 
    
        
        
        
        
        
        
        
        
        