import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from folium import plugins
import numpy as np
from sklearn.linear_model import ElasticNet
from scipy.optimize import curve_fit
import plotly.express as px
from apps import z_functions as zf

def app():
    # title of the app
       
    st.markdown('Artillery Physics')
    c1,c2 = st.columns((1,3))
    with c1:
        galt = int(st.text_input('Gun Altitude (Meters):',700))
        imv = st.text_input('Initial Muzzle Velocity (m/s):',500)
        qe = int(st.text_input('Tube Elevation (mils):',250))
        sw = int(st.text_input('Shell Weight (lbs):',103.5))
        m = sw/2.20462 #mass lbs to kg
        th0 = qe * np.pi / 3200 # initial angle in radians
        x0 = 0 #Initial x
        y0 = galt # initial y
        t0 = 0 # initial time
        g = 9.80665 # gravitational force in m/s/s
        
        alt_press = pd.DataFrame({'alt':[0,200,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,6000,7000,8000,9000],
                           'rho':[1.2250,1.2133,1.1844,1.1392,1.0846,1.0320,.9569,.8632,.7768,.6971,.5895,.4664,.3612,.2655,.1937,.1413]})
                            #Air Pressure at differnet altitudes
        press_M = np.poly1d(np.polyfit(alt_press['alt'], alt_press['rho'], 5)) #Pressure Model for any altitude
        alts = np.arange(0, 9000, 100)
        press = press_M(alts)
        fig = px.scatter(alt_press, x=alts, y=press)
        st.plotly_chart(fig)
    with c2:
        pk = .000006 * 46.94681 #This is my predicted k value for the shape of the round.
        rho0 = press_M(galt) #initial Air Pressure
        k = rho0*pk/m # is the drag constant at 0 alt
        
        
        data = pd.DataFrame({'Initial Air Pressure':str(rho0)},index = ['Fire Mission']).T 
        st.dataframe(data,height=500) 
