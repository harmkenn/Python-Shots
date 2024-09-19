import streamlit as st
import folium
from streamlit_folium import st_folium

# Create a Folium map
m = folium.Map(location=[45.5236, -122.6750], zoom_start=13)

# Add LatLngPopup to the map
folium.LatLngPopup().add_to(m)

# Display the map in Streamlit
st_folium(m, width=700, height=500)

# Create a text input field
user_input = st.text_area("Paste the text you want to copy here")



# Display the user's input
if user_input:
    st.write("You entered:")
    st.write(user_input)