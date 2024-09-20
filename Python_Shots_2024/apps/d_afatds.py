import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import plugins
from numpy import radians, sin, exp, cos, tan, sqrt, arctan, pi, log
import matplotlib.pyplot as plt
from apps import z_functions as zf
import re
import plotly.express as px

def app():
    # title of the app
    lu = st.sidebar.text_input('Lookup: ') 
    if len(lu)>=3:
        where = zf.lookup(lu)
        st.sidebar.write(where[0])
        st.sidebar.write('Lat: '+str(where[1])+' Lon: '+str(where[2]))
        st.sidebar.write('MGRS: '+where[3])
        alt = zf.elevation(where[1],where[2])
        st.sidebar.write('Alt :'+str(round(alt,2))+' Meters')
    #st.markdown('Physics Shot')

    c1,c2,c3 = st.columns((1,3,1))

    with c1:
        if 'na_lpmgrs' not in st.session_state: st.session_state['na_lpmgrs'] = '36SYC6482056356'
        na_lpmgrs = st.session_state['na_lpmgrs']
        if 'na_ipmgrs' not in st.session_state: st.session_state['na_ipmgrs'] = '36SYC6482066356'
        na_ipmgrs = st.session_state['na_ipmgrs']
        # Create a text input field
        user_input = st.text_area("Paste Point", height=24)
        # Create the button
        rad = st.radio('lat lon to MGRS',['to Launch Point','to Impact Point'])
        if st.button("convert"):
            if rad == 'to Launch Point':
                # Split the string by the newline character
                parts = user_input.split('\n')
                # The first part contains the latitude
                latitude_part = parts[0]
                # Extract the numeric value after 'Latitude: '
                latitude = float(latitude_part.split(': ')[1])
                longitude_part = parts[1]
                longitude = float(longitude_part.split(': ')[1])
                na_lpmgrs = zf.LL2MGRS(latitude,longitude)[1]
                
            if rad == 'to Impact Point':
                # Split the string by the newline character
                parts = user_input.split('\n')
                # The first part contains the latitude
                latitude_part = parts[0]
                # Extract the numeric value after 'Latitude: '
                latitude = float(latitude_part.split(': ')[1])
                longitude_part = parts[1]
                longitude = float(longitude_part.split(': ')[1])
                na_ipmgrs = zf.LL2MGRS(latitude,longitude)[1]
        
        na_lpmgrs = st.text_input('Launch Point (MGRS):',na_lpmgrs, key = 'c1')
        lp = zf.MGRS2LL(na_lpmgrs)
        l_lat = lp[1] #Launch Latitude
        l_lon = lp[2] #Launch Longitude
        lp_alt = zf.elevation(l_lat,l_lon) #Launch Altitude in meters
        #st.write(lp_alt)
        lp_alt = float(st.text_input('launch Altitude (meters)',lp_alt))
        
        na_ipmgrs = st.text_input('Impact Point (MGRS):',na_ipmgrs, key = 'c2')
        ip = zf.MGRS2LL(na_ipmgrs)
        i_lat = ip[1] #Launch Latitude
        i_lon = ip[2] #Launch Longitude
        ip_alt = zf.elevation(i_lat,i_lon) #Launch Altitude in meters
        #st.write(lp_alt)
        ip_alt = float(st.text_input('Impact Altitude (meters)',ip_alt))
        st.session_state['na_lpmgrs'] = na_lpmgrs
        st.session_state['na_ipmgrs'] = na_ipmgrs
        range = zf.LLDist(l_lat,l_lon,i_lat,i_lon)[0]
        
        mass = st.text_input('Mass (lbs)',103.5)
        
        
        if range <= 12000:
            mv = st.text_input('Initial Velocity (m/s)',int(-2.305882*float(mass)+806.6588))
        elif range > 12000 and range <= 18000:
            mv = st.text_input('Initial Velocity (m/s)',int(-2.105882*float(mass)+899.9588))
        elif range > 18000:
            mv = st.text_input('Initial Velocity (m/s)',int(-2.447059*float(mass)+1043.271))

        with c3:
            aol = int(st.text_input('Azimuth of Lay ', 2200))
            st.write('Range: ' + str(int(range)) + ' m')
            # Inputs
            # Example usage:
            HorL = st.radio('High or Low Angle',['High','Low'])
            m_kg = float(mass)/2.2 #kg
            drag_coefficient = .25
            diam_mm = 155
            r_m = float(diam_mm)/2000 #radius in meters
            area = r_m**2* pi #Front round area in square meters
            az_degree = zf.LLDist(l_lat,l_lon,i_lat,i_lon)[1]
            
            if HorL == 'High': out = zf.MLAHshot(lp[1],lp_alt,ip_alt,mass,az_degree/180*3200,aol,mv,range)
            if HorL == 'Low': out = zf.MLALshot(lp[1],lp_alt,ip_alt,mass,az_degree/180*3200,aol,mv,range)
            st.write('Max Ord (m): '+str(int(out[1])))
            st.write('Time of Flight (s): '+str(int(out[2])))
            st.write('QE (mils): '+str(int(out[0])))
            st.write('Deflection (mils): '+str(int(out[4])))
            st.write('Drift (mils): '+str(int(out[3])))
            
            # Additional input for Magnus effect
            mv = float(mv)
            spin_rate = mv/(20*float(diam_mm)/1000) # rifling based on 20 calibers per 1 spin
            spin_velocity = spin_rate * 2 * pi  # Convert from rev/s to rad/
            # Call the function and display the  physics results
            la_degree = out[0]/3200*180
            
            time_of_flight, final_x, final_y, final_z, t_time, x_vals, y_vals, z_vals, vx_comp, vy_comp,vz_comp, v_mag,qe = zf.projectile_motion_with_drag(mv, la_degree, m_kg, drag_coefficient, area, l_lat, az_degree, lp_alt, lp_alt, spin_velocity)
            shotdata = pd.DataFrame({'t':t_time,'x':x_vals,'y':y_vals,'z':z_vals, 'vx':vx_comp, 'vy':vy_comp,'vz':vz_comp, 'QE':qe,'v_mag':v_mag})
            shotdata2 = shotdata
            shotdata2['t2'] = shotdata['t']/(max(shotdata['t']))*out[2]
            shotdata2['x2'] = shotdata['x']/(max(shotdata['x']))*range
            shotdata2['y2'] = shotdata['y']-lp_alt 
            shotdata2['y2'] = shotdata2['y2']/(max(shotdata2['y2']))*(out[1]-lp_alt) 
            shotdata2['y2'] = shotdata2['y2']+lp_alt 
            shotdata2['drift'] = arctan(shotdata['z']/shotdata['x'])/pi*3200
            wdrift = max(shotdata2['drift'][-5:])
            shotdata2['drift'] = shotdata2['drift']/wdrift*int(out[3])
            shotdata2['z2'] = shotdata2['x2']*tan(shotdata2['drift']/3200*pi)
            shotdata2['dx'] =  shotdata2['x2'].diff()
            shotdata2['dy'] =  shotdata2['y2'].diff()
            shotdata2['dt'] =  shotdata2['t2'].diff()
            shotdata2['qe2'] = arctan(shotdata2['dy']/shotdata2['dx'])/pi*3200
            shotdata2['v2'] = sqrt(shotdata2['dy']**2+shotdata2['dx']**2)/shotdata2['dt']

            st.write('Final Velocity (m/s): '+str(int(shotdata2['v2'].iat[-10])))
            st.write('Final QE (mils): '+str(int(shotdata2['qe2'].iloc[-10])))
            deets = zf.vPolar(float(lp[1]),float(lp[2]),az_degree+int(out[3])/3200*180,range)
            st.write('Launch Bearing: '+str(round(az_degree,2)) + ' degrees')
            st.write('Launch Azimuth: '+str(round(az_degree/180*3200,2)) + ' mils')
            st.write('Impact Bearing: '+str(round(deets[2],2)) + ' degrees')
            st.write('Impact Azimuth: '+str(round(deets[2]*3200/180,2)) + ' mils')
    
        with c2:
            # map 
            map = folium.Map(location=[lp[1],lp[2]], zoom_start=-1.36*log(30000/1000)+15)
            # Add a circle to the center of the map
            folium.Circle(location=[l_lat, l_lon],radius=20000,color='red',
                fill=True,fill_color='red',fill_opacity=0.1).add_to(map)
            # add tiles to map
            attribution = "Map tiles by Google"
            folium.raster_layers.TileLayer('Open Street Map', attr=attribution).add_to(map)
            folium.raster_layers.TileLayer('Stamen Terrain', attr=attribution).add_to(map)
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
                    tiles = 'https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
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
            pal = folium.features.CustomIcon('Python_Shots_2024/Icons/paladin.jpg',icon_size=(30,20))
            tgt = folium.features.CustomIcon('Python_Shots_2024/Icons/target.png',icon_size=(25,25))
            
            folium.Marker(location=[lp[1],lp[2]], color='green',popup=na_lpmgrs, tooltip='Launch Point',icon=pal).add_to(map)
            folium.Marker(location=[ip[1],ip[2]], color='green',popup=na_ipmgrs, tooltip='Impact Point',icon=tgt).add_to(map)
            folium.PolyLine(locations=[[lp[1],lp[2]], [ip[1],ip[2]]], color='red', weight=3).add_to(map)
            oncircle = zf.vPolar(lp[1],lp[2],aol/3200*180,20000)
            folium.PolyLine(locations=[[lp[1],lp[2]], [oncircle[0],oncircle[1]]], color='blue', weight=3).add_to(map)
            map.add_child(folium.LatLngPopup())

            folium_static(map)

            fig = px.line(shotdata2, x='x2', y='y2', title='Drag, Varying Air Density, Varying Gravity, Initial Altitude', labels={'x2': 'Distance (m)', 'y2': 'Height (m)'})
            fig.update_layout(xaxis=dict(tickfont=dict(size=8)), yaxis=dict(tickfont=dict(size=8)), legend=dict(font=dict(size=8)), width=700, height=300, margin=dict(l=20, r=20, t=40, b=20))
            fig.update_xaxes(range=[0,range])
            fig.update_yaxes(range=[0,int(out[1])])
            #fig.update_xaxes(matches='y')
            fig.update_yaxes(matches='x')
           

            st.plotly_chart(fig, use_container_width=True)
                                                                                                                                  
