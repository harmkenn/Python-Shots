import streamlit as st
import pandas as pd

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