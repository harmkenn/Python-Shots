import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import plugins

from apps import z_functions as zf

# Function to read Excel file and extract sheet names
def get_excel_sheets(file):
    # Use pandas ExcelFile to read the Excel file
    xls = pd.ExcelFile(file)
    # Get the list of sheet names
    sheet_names = xls.sheet_names
    return sheet_names

def app():
    # title of the app
    lu = st.sidebar.text_input('Lookup: ') 
    go = 0
    if len(lu)>=3:
        where = zf.lookup(lu)
        st.sidebar.write(where[0])
        st.sidebar.write('Lat: '+str(where[1])+' Lon: '+str(where[2]))
        st.sidebar.write('MGRS: '+where[3])
        alt = zf.elevation(where[1],where[2])
        st.sidebar.write('Alt :'+str(round(alt,2))+' Meters')

    c1,c2 = st.columns((1,2))
    with c1:
        if 'a_ipmgrs' not in st.session_state: st.session_state['a_ipmgrs'] = '29RPR6518104660'
        ipmgrs = st.session_state['a_ipmgrs']
        ipmgrs = st.text_input('Impact Point (MGRS):','29RPR6518104660')
        st.session_state['a_ipmgrs'] = ipmgrs
      
        ip = zf.MGRS2LL(ipmgrs)
        st.write('(LL): '+str(round(ip[1],3))+', '+str(round(ip[2],5)))

        # File uploader to allow users to upload an Excel file
        uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])
        # If file is uploaded
        if uploaded_file is not None:
          
            # Read Excel file and extract sheet names
            sheet_names = get_excel_sheets(uploaded_file)
            # Display list of sheet names
            selected_sheet = st.selectbox("Select a sheet:", sheet_names)
            if selected_sheet:
                # Read selected sheet from Excel file into pandas dataframe
                df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        
                # Create dropdown boxes for selecting columns
                selected_name_column = st.selectbox("Select Name column:", df.columns)
                selected_description_column = st.selectbox("Select Description column:", df.columns)
                selected_type_column = st.selectbox("Select Type column:", df.columns)
                selected_lat_column = st.selectbox("Latitude:", df.columns)
                selected_lon_column = st.selectbox("Longitude:", df.columns)
                
                if st.button("Process"):
                    # Create a new dataframe with selected columns
                    add_df = df[[selected_name_column, selected_description_column, selected_type_column, selected_lat_column, selected_lon_column]]
                    go = 1
                    # Display the resulting dataframe
                    st.write(add_df)
                    
        else:
            st.write("Upload an Excel file above.")
            
        

    with c2:
        # map
        map = folium.Map(location=[ip[1], ip[2]], zoom_start=5)
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
        safe = folium.features.CustomIcon('Python_Shots_2024/Icons/safe.png',icon_size=(25,25))

        folium.Marker(location=[ip[1],ip[2]], color='green',popup=ipmgrs, tooltip='Impact Point',icon=tgt).add_to(map)
        add_df = add_df.dropna()
        for index, row in add_df.iterrows():
            folium.CircleMarker(location=[row['lat'], row['long']],tooltip=row['Facility Name'] ,radius=3, color='blue', fill=True, fill_color='blue').add_to(map)
    
    
        
        
        draw = plugins.Draw()
        draw.add_to(map)
        # display map
        folium_static(map) 

        st.write(add_df.sort_values(by='dist'))
        

            
    
        
        
        
        
        
        
        
        
        