import os
import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import folium
from folium import plugins
import pandas as pd

# Import helper functions module
from SubApps import z_functions as zf

# 1. Register page in Dash registry
dash.register_page(
    __name__,
    path="/",  # Default home route or change to '/deflection'
    name="Deflection",
    order=1
)

# 2. Page Layout definition
layout = dbc.Container(
    [
        html.H2("Deflection", className="mb-4"),
        dbc.Row(
            [
                # Left Column (Controls & Inputs)
                dbc.Col(
                    [
                        # Coordinates Block
                        dbc.Card(
                            [
                                dbc.CardHeader("Coordinates Lookup & Conversion"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Latitude"),
                                                        dbc.Input(id="def-lat", type="number", value=-30, step="any"),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Longitude"),
                                                        dbc.Input(id="def-lon", type="number", value=30, step="any"),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        html.Div(id="def-ll-converted-output", className="mb-3 text-muted"),
                                        html.Hr(),
                                        dbc.Label("Lookup Identifier"),
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(id="def-lookup", type="text", placeholder="Enter a place or address..."),
                                                dbc.Button("Search", id="def-lookup-button", n_clicks=0),
                                            ]
                                        ),
                                        html.Div(id="def-lookup-output", className="mt-2 text-info"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        # Tactical Inputs Block
                        dbc.Card(
                            [
                                dbc.CardHeader("Tactical Parameters"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Azimuth of Fire (mils):"),
                                        dbc.Input(id="def-aof", type="number", value=2000, className="mb-2"),
                                        
                                        dbc.Label("Start Point (MGRS):"),
                                        dbc.Input(id="def-lpmgrs", type="text", value="", className="mb-2"),
                                        html.Div(id="def-start-ll-output", className="text-muted mb-3"),
                                        
                                        dbc.Label("End Point (MGRS):"),
                                        dbc.Input(id="def-ipmgrs", type="text", value="", className="mb-3"),
                                        
                                        html.Div(id="def-tp-output", className="text-muted"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                    ],
                    width=4,
                ),
                # Right Column (Map & Tactical Calculations)
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Interactive Tactical Map"),
                                dbc.CardBody(
                                    [
                                        # Embedded Folium Map Container
                                        html.Iframe(
                                            id="def-map-iframe",
                                            style={"width": "100%", "height": "500px", "border": "none"},
                                        )
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Fire Control Calculations"),
                                dbc.CardBody(html.Div(id="def-details-output")),
                            ]
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
    ],
    fluid=True,
)


# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------

# Callback 1: Automatically set MGRS inputs when Lat/Lon changes
@callback(
    Output("def-ll-converted-output", "children"),
    Output("def-lpmgrs", "value"),
    Output("def-ipmgrs", "value"),
    Input("def-lat", "value"),
    Input("def-lon", "value"),
)
def update_initial_mgrs(lat, lon):
    if lat is None or lon is None:
        return "", "", ""
    try:
        back = zf.LL2MGRS(float(lat), float(lon))
        ip = zf.LL2MGRS(float(lat) + 0.1, float(lon) + 0.1)

        info_text = html.Div(
            [
                html.P(f"UTM: {back[0]}", className="mb-1"),
                html.P(f"MGRS: {back[1]}", className="mb-0"),
            ]
        )
        return info_text, back[1], ip[1]
    except Exception as e:
        return f"Conversion error: {e}", "", ""


@callback(
    Output("def-start-ll-output", "children"),
    Input("def-lpmgrs", "value"),
)
def display_start_point_coordinates(lpmgrs):
    if not lpmgrs:
        return ""
    try:
        start = zf.MGRS2LL(lpmgrs)
        return f"Start Point (LL): {round(start[1], 5)}, {round(start[2], 5)}"
    except Exception as e:
        return f"Invalid MGRS: {e}"


# Callback 2: Optional Lookup functionality
@callback(
    Output("def-lookup-output", "children"),
    Input("def-lookup-button", "n_clicks"),
    State("def-lookup", "value"),
)
def process_lookup(n_clicks, lu):
    if not n_clicks or not lu or not lu.strip():
        return ""
    try:
        where = zf.lookup(lu)
        alt = zf.elevation(where[1], where[2])
        return html.Div(
            [
                html.P(where[0], className="mb-0 fw-bold"),
                html.P(f"Lat: {where[1]} | Lon: {where[2]}", className="mb-0"),
                html.P(f"MGRS: {where[3]}", className="mb-0"),
                html.P(f"Alt: {round(alt, 2)} Meters", className="mb-0"),
            ]
        )
    except Exception as e:
        return f"Lookup error: {e}"


# Callback 3: Update Map & Fire Control Metrics
@callback(
    Output("def-map-iframe", "srcDoc"),
    Output("def-tp-output", "children"),
    Output("def-details-output", "children"),
    Input("def-aof", "value"),
    Input("def-lpmgrs", "value"),
    Input("def-ipmgrs", "value"),
)
def update_map_and_metrics(aof, lpmgrs, ipmgrs):
    if not lpmgrs or not ipmgrs or aof is None:
        return "", "", ""

    try:
        aof = float(aof)
        lp = zf.MGRS2LL(lpmgrs)
        ip = zf.MGRS2LL(ipmgrs)

        # Tactical Point Info Output
        tp_info = html.Div(
            [
                html.P(f"UTM: {ip[0]}", className="mb-1"),
                html.P(f"End Point (LL): {round(ip[1], 5)}, {round(ip[2], 5)}", className="mb-0"),
            ]
        )

        # Generate Map using Folium
        fmap = folium.Map(location=[lp[1], lp[2]], zoom_start=10)

        # Layers
        folium.TileLayer('OpenStreetMap').add_to(fmap)
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Maps',
            overlay=False
        ).add_to(fmap)
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=False
        ).add_to(fmap)
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Terrain',
            overlay=False
        ).add_to(fmap)

        folium.LayerControl().add_to(fmap)
        fmap.add_child(plugins.MiniMap(toggle_display=True))
        plugins.ScrollZoomToggler().add_to(fmap)
        plugins.Fullscreen(position='topright').add_to(fmap)

        # Resolve icons relative to this app, so launching Dash from another
        # working directory does not break the marker images.
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        icons_dir = os.path.join(app_dir, 'Icons')
        pal_path = os.path.join(icons_dir, 'paladin.jpg')
        tgt_path = os.path.join(icons_dir, 'target.png')

        # Icons & Markers for Start Point (lp) and End Point (ip)
        if os.path.exists(pal_path) and os.path.exists(tgt_path):
            pal = folium.features.CustomIcon(pal_path, icon_size=(30, 20))
            tgt = folium.features.CustomIcon(tgt_path, icon_size=(25, 25))
            
            folium.Marker(
                location=[lp[1], lp[2]], 
                popup=lpmgrs, 
                tooltip='Start Point', 
                icon=pal
            ).add_to(fmap)
            
            folium.Marker(
                location=[ip[1], ip[2]], 
                popup=ipmgrs, 
                tooltip='End Point', 
                icon=tgt
            ).add_to(fmap)
        else:
            print(f"Icon files not found at:\n - {pal_path}\n - {tgt_path}")
            folium.Marker(
                location=[lp[1], lp[2]], 
                popup=lpmgrs, 
                tooltip='Start Point', 
                icon=folium.Icon(color='green')
            ).add_to(fmap)
            
            folium.Marker(
                location=[ip[1], ip[2]], 
                popup=ipmgrs, 
                tooltip='End Point', 
                icon=folium.Icon(color='red')
            ).add_to(fmap)

        # Polylines & Range Rings
        pback = zf.vPolar(lp[1], lp[2], aof * 180 / 3200, 30000)
        folium.PolyLine([[lp[1], lp[2]], [pback[0], pback[1]]], tooltip='Azimuth of Fire', popup=str(aof)).add_to(fmap)
        folium.PolyLine([[lp[1], lp[2]], [ip[1], ip[2]]], color='red', tooltip='Gun Target Line', popup='GTL').add_to(fmap)
        folium.Circle(radius=30000, location=[lp[1], lp[2]], color='green').add_to(fmap)

        plugins.Draw().add_to(fmap)

        # Convert Folium map to HTML string for the Iframe container
        map_html = fmap._repr_html_()

        # Calculation Details
        deets = zf.LLDist(lp[1], lp[2], ip[1], ip[2])
        bearing_deg = deets[1]
        azimuth_mils = bearing_deg * 3200 / 180
        diff = round(aof - azimuth_mils + 3200, 0)
        if diff < 0:
            diff += 6400

        details = dbc.Row(
            [
                dbc.Col([html.Strong("Distance:"), html.Span(f" {int(deets[0])} meters")], width=6),
                dbc.Col([html.Strong("Bearing:"), html.Span(f" {round(bearing_deg, 2)}°")], width=6),
                dbc.Col([html.Strong("Azimuth:"), html.Span(f" {round(azimuth_mils, 2)} mils")], width=6),
                dbc.Col([html.Strong("Deflection:"), html.Span(f" {diff} mils", className="text-danger fw-bold")], width=6),
            ],
            className="g-2",
        )

        return map_html, tp_info, details

    except Exception as e:
        return "", f"Error computing metrics: {e}", ""
