import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import plugins
from numpy import radians, sin, exp, cos, tan, sqrt, arctan, pi, log
import matplotlib.pyplot as plt
from apps import z_functions as zf

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
    st.markdown('Physics Shot')

    c1,c2,c3 = st.columns((1,3,1))

    with c1:
        
        if 'c_lpmgrs' not in st.session_state: st.session_state['c_lpmgrs'] = '36SYC6482056356'
        c_lpmgrs = st.session_state['c_lpmgrs']
        c_lpmgrs = st.text_input('Launch Point (MGRS):',c_lpmgrs, key = 'c1')
        st.session_state['c_lpmgrs'] = c_lpmgrs
        lp = zf.MGRS2LL(c_lpmgrs)
        l_lat = lp[1] #Launch Latitude
        l_lon = lp[2] #Launch Longitude
        lp_alt = zf.elevation(l_lat,l_lon) #Launch Altitude in meters
        #st.write(lp_alt)
        lp_alt = float(st.text_input('launch Altitude (meters)',lp_alt))
        
        # Inputs
        # Example usage:
        v0 = float(st.text_input('Initial Velocity (m/s)',810.8)) #m/s
        la_mils = st.text_input('Launch Angle (mils)',357.4) #mils
        la_degree = float(la_mils)/3200*180 #mils to degrees
        m_lbs = st.text_input('Mass (lbs)',95) #lbs
        m_kg = float(m_lbs)/2.2 #kg
        drag_coefficient = .25
        diam_mm = st.text_input('Diameter (mm)',155) #mm
        r_m = float(diam_mm)/2000 #radius in meters
        area = r_m**2* pi #Front round area in square meters
        az_mils = st.text_input(' Gun-Target Azimuth (mils)',5731) #
        az_degree = float(az_mils)/3200*180
        aol = st.text_input('Azimuth of Lay (mils)',1234) #
        
        # Additional input for Magnus effect
        spin_rate = float(v0)/(20*float(diam_mm)/1000) # rifling based on 20 calibers per 1 spin
        spin_velocity = spin_rate * 2 * pi  # Convert from rev/s to rad/
        # Call the function and display the  physics results
        time_of_flight, final_x, final_y, final_z, t_time, x_vals, y_vals, z_vals, vx_comp, vy_comp,vz_comp, v_mag,qe = zf.projectile_motion_with_drag(v0, la_degree, m_kg, drag_coefficient, area, l_lat, az_degree, lp_alt, lp_alt, spin_velocity)
    with c3:
        #compute the Machine learning results
        out = zf.MLshot(l_lat,lp_alt,lp_alt,m_lbs,az_mils,aol,la_mils,v0)
        st.write('ML output:')
        st.write('Range (m): '+str(int(out[0])))
        st.write('Max Ord (m): '+str(int(out[1])))
        st.write('Time of Flight (s): '+str(int(out[2])))
        
        st.write('Deflection (mils): '+str(int(out[4])))
        st.write('Drift (mils): '+str(int(out[3])))
       
        shotdata = pd.DataFrame({'t':t_time,'x':x_vals,'y':y_vals,'z':z_vals, 'vx':vx_comp, 'vy':vy_comp,'vz':vz_comp, 'QE':qe,'v_mag':v_mag})
        shotdata2 = shotdata
        shotdata2['t2'] = shotdata['t']/(max(shotdata['t']))*out[2]
        shotdata2['x2'] = shotdata['x']/(max(shotdata['x']))*out[0]
        shotdata2['y2'] = shotdata['y']-lp_alt 
        shotdata2['y2'] = shotdata2['y2']/(max(shotdata2['y2']))*(out[1]-lp_alt) 
        shotdata2['y2'] = shotdata2['y2']+lp_alt 
        drift = out[3]
        shotdata2['drift'] = arctan(shotdata['z']/shotdata['x'])/pi*3200
        wdrift = max(shotdata2['drift'][-5:])
        shotdata2['drift'] = shotdata2['drift']/wdrift*drift
        shotdata2['z2'] = shotdata2['x2']*tan(shotdata2['drift']/3200*pi)
        shotdata2['dx'] =  shotdata2['x2'].diff()
        shotdata2['dy'] =  shotdata2['y2'].diff()
        shotdata2['dt'] =  shotdata2['t2'].diff()
        shotdata2['qe2'] = arctan(shotdata2['dy']/shotdata2['dx'])/pi*3200
        shotdata2['v2'] = sqrt(shotdata2['dy']**2+shotdata2['dx']**2)/shotdata2['dt']
        st.write('Drift (m): '+str(int(shotdata2['z2'].iat[-10])))
        
        

        

        
    
    if len(c_lpmgrs)>3 and len(az_mils)>1:
        lp = zf.MGRS2LL(c_lpmgrs)
        l_azdeg = (float(az_mils))*180/3200
        final_x = float(final_x)
        with c3:
            st.write('Final Velocity (m/s): '+str(int(shotdata2['v2'].iat[-10])))
            st.write('Final QE (mils): '+str(int(shotdata2['qe2'].iloc[-10])))
            deets = zf.vPolar(float(lp[1]),float(lp[2]),l_azdeg+drift/3200*180,final_x)
            st.write('Launch Bearing: '+str(round(l_azdeg,2)) + ' degrees')
            st.write('Launch Azimuth: '+str(round(float(az_mils),2)) + ' mils')
            st.write('Impact Bearing: '+str(round(deets[2],2)) + ' degrees')
            st.write('Impact Azimuth: '+str(round(deets[2]*3200/180,2)) + ' mils')
            ip = zf.LL2MGRS(deets[0],deets[1])
            st.write('Launch Point (LL): '+str(round(l_lat,5))+', '+str(round(l_lon,5))+', '+str(round(lp_alt,0)))
            st.write('Impact Location (MGRS): '+ip[1])
            st.write('Impact Point (LL): '+str(round(deets[0],5))+', '+str(round(deets[1],5)))
        with c2:            
            # map
            if deets[1]>lp[2] and l_azdeg > 180: deets[1] = deets[1] - 360
            if deets[1]<lp[2] and l_azdeg < 180: deets[1] = deets[1] + 360
            map = folium.Map(location=[(lp[1]+deets[0])/2, (lp[2]+deets[1])/2], zoom_start=-1.36*log(final_x/1000)+15)
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
            pal = folium.features.CustomIcon('Icons/paladin.jpg',icon_size=(30,20))
            tgt = folium.features.CustomIcon('Icons/target.png',icon_size=(25,25))
            
            folium.Marker(location=[lp[1],lp[2]], color='green',popup=c_lpmgrs, tooltip='Launch Point',icon=pal).add_to(map)
            folium.Marker(location=[deets[0],deets[1]], color='green',popup=ip[1], tooltip='Impact Point',icon=tgt).add_to(map)
            

            
        with c2:

             # st.write(deets)
            points = []
            points.append([lp[1],lp[2]])
            td = l_azdeg
            
            for p in range(0,1000):
                get = zf.vPolar(points[p][0],points[p][1],td,final_x/1000)
                if l_azdeg > 180 and points[p][1] > points [p-1][1]: points[p][1] = points[p][1] - 360
                points.append([get[0],get[1]])
                td = zf.LLDist(get[0],get[1],deets[0],deets[1])[1]
            
            del points[-1]
            points.append([deets[0],deets[1]])
            folium.PolyLine(points, color='red').add_to(map)
            
            draw = plugins.Draw()
            draw.add_to(map)
            # display map
            folium_static(map) 

            # Plotting the trajectory
            fig, ax = plt.subplots(figsize=(7, 3))
            plt.plot(shotdata2['x2'],shotdata2['y2'], label='Projectile Trajectory')
            plt.title('Drag, Varying Air Density, Varying Gravity, Initial Altitude')
            plt.xlabel('Distance (m)')
            plt.ylabel('Height (m)')
            plt.legend()
            plt.grid(True)
            # Set equal aspect ratio for x and y axes
            ax.set_aspect('equal', adjustable='box')
            st.pyplot(fig)
            st.write(shotdata2.iloc[:, -5:])


            
            


        
    