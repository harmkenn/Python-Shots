"""AFATDS ballistic planning page for the Dash Shots app."""

import math
import os
import re

import dash
from dash import Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc
import folium
import plotly.express as px
import plotly.graph_objects as go
from folium import plugins

from SubApps import z_functions as zf


dash.register_page(__name__, path="/d_afatds", name="AFATDS", order=4)


layout = dbc.Container(
    [
        html.H2("AFATDS Shot Planning", className="mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Location lookup"),
                                dbc.CardBody(
                                    [
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(
                                                    id="af-lookup",
                                                    type="text",
                                                    placeholder="Place or address (3+ characters)",
                                                ),
                                                dbc.Button("Lookup", id="af-lookup-button", n_clicks=0),
                                            ],
                                            className="mb-2",
                                        ),
                                        html.Div(id="af-lookup-output"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Shot coordinates"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Paste point coordinates"),
                                        dbc.Textarea(
                                            id="af-paste-point",
                                            placeholder="Latitude: 00.0000\nLongitude: 00.0000",
                                            style={"height": "68px"},
                                            className="mb-2",
                                        ),
                                        dbc.RadioItems(
                                            id="af-paste-destination",
                                            options=[
                                                {"label": "Convert to launch point", "value": "launch"},
                                                {"label": "Convert to impact point", "value": "impact"},
                                            ],
                                            value="launch",
                                            inline=True,
                                            className="mb-2",
                                        ),
                                        dbc.Button(
                                            "Convert coordinates",
                                            id="af-convert-button",
                                            n_clicks=0,
                                            color="secondary",
                                            outline=True,
                                            className="mb-3",
                                        ),
                                        html.Div(id="af-convert-output", className="mb-2"),
                                        dbc.Label("Launch Point (MGRS)"),
                                        dbc.Input(
                                            id="af-launch-mgrs",
                                            type="text",
                                            value="36SYC6482056356",
                                            className="mb-2",
                                        ),
                                        dbc.Label("Launch Altitude (meters)"),
                                        dbc.Input(id="af-launch-altitude", type="number", value=0, step="any", className="mb-2"),
                                        dbc.Label("Impact Point (MGRS)"),
                                        dbc.Input(
                                            id="af-impact-mgrs",
                                            type="text",
                                            value="36SYC6482066356",
                                            className="mb-2",
                                        ),
                                        dbc.Label("Impact Altitude (meters)"),
                                        dbc.Input(id="af-impact-altitude", type="number", value=0, step="any", className="mb-2"),
                                        html.Div(id="af-coordinate-output", className="text-muted"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Projectile parameters"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Mass (lbs)"),
                                        dbc.Input(id="af-mass", type="number", value=103.5, min=0, step="any", className="mb-2"),
                                        dbc.Label("Initial Velocity (m/s)"),
                                        dbc.Input(id="af-velocity", type="number", value=568, min=0, step="any", className="mb-2"),
                                        dbc.Label("Azimuth of Lay (mils)"),
                                        dbc.Input(id="af-aol", type="number", value=2200, step="any", className="mb-2"),
                                        dbc.Label("Trajectory"),
                                        dbc.RadioItems(
                                            id="af-angle-mode",
                                            options=[{"label": "High angle", "value": "high"}, {"label": "Low angle", "value": "low"}],
                                            value="high",
                                            inline=True,
                                        ),
                                        html.Div(id="af-range-output", className="mt-3 fw-bold"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Shot solution"),
                                dbc.CardBody(html.Div(id="af-solution-output")),
                            ]
                        ),
                    ],
                    xs=12,
                    xl=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Shot map"),
                                dbc.CardBody(
                                    dcc.Loading(
                                        html.Iframe(
                                            id="af-map-iframe",
                                            title="Map showing the launch point, impact point, and azimuth of lay",
                                            style={"width": "100%", "height": "560px", "border": "none"},
                                        )
                                    )
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Projectile trajectory"),
                                dbc.CardBody(dcc.Graph(id="af-trajectory-graph", figure=go.Figure())),
                            ],
                        ),
                    ],
                    xs=12,
                    xl=8,
                ),
            ],
            className="g-3",
        ),
    ],
    fluid=True,
)


@callback(
    Output("af-lookup-output", "children"),
    Input("af-lookup-button", "n_clicks"),
    State("af-lookup", "value"),
    prevent_initial_call=True,
)
def process_lookup(n_clicks, query):
    if not n_clicks or not query or len(query.strip()) < 3:
        return "Enter at least three characters to search."
    try:
        place = zf.lookup(query.strip())
        altitude = zf.elevation(place[1], place[2])
        return html.Div(
            [
                html.P(place[0], className="mb-1 fw-bold"),
                html.P(f"Lat: {place[1]} | Lon: {place[2]}", className="mb-1"),
                html.P(f"MGRS: {place[3]}", className="mb-1"),
                html.P(f"Alt: {round(altitude, 2)} meters", className="mb-0"),
            ]
        )
    except Exception as exc:
        return dbc.Alert(f"Lookup error: {exc}", color="danger", className="mb-0")


@callback(
    Output("af-launch-mgrs", "value"),
    Output("af-impact-mgrs", "value"),
    Output("af-convert-output", "children"),
    Input("af-convert-button", "n_clicks"),
    State("af-paste-point", "value"),
    State("af-paste-destination", "value"),
    State("af-launch-mgrs", "value"),
    State("af-impact-mgrs", "value"),
    prevent_initial_call=True,
)
def convert_pasted_coordinates(n_clicks, pasted, destination, launch_mgrs, impact_mgrs):
    if not n_clicks:
        return launch_mgrs, impact_mgrs, ""
    if not pasted:
        return launch_mgrs, impact_mgrs, dbc.Alert("Paste latitude and longitude first.", color="warning")

    # Accept both the labeled two-line text used by the source app and a plain
    # pair of decimal coordinates separated by commas or whitespace.
    values = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", pasted)
    if len(values) < 2:
        return launch_mgrs, impact_mgrs, dbc.Alert("Could not find latitude and longitude in the pasted text.", color="danger")
    try:
        latitude, longitude = map(float, values[:2])
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("Latitude or longitude is outside its valid range.")
        mgrs = zf.LL2MGRS(latitude, longitude)[1]
        if destination == "impact":
            return launch_mgrs, mgrs, f"Impact point converted to {mgrs}."
        return mgrs, impact_mgrs, f"Launch point converted to {mgrs}."
    except Exception as exc:
        return launch_mgrs, impact_mgrs, dbc.Alert(f"Coordinate conversion error: {exc}", color="danger")


@callback(
    Output("af-launch-altitude", "value"),
    Input("af-launch-mgrs", "value"),
)
def set_default_launch_altitude(mgrs):
    if not mgrs:
        return None
    try:
        point = zf.MGRS2LL(mgrs)
        return round(float(zf.elevation(point[1], point[2])), 2)
    except Exception:
        return None


@callback(
    Output("af-impact-altitude", "value"),
    Input("af-impact-mgrs", "value"),
)
def set_default_impact_altitude(mgrs):
    if not mgrs:
        return None
    try:
        point = zf.MGRS2LL(mgrs)
        return round(float(zf.elevation(point[1], point[2])), 2)
    except Exception:
        return None


@callback(
    Output("af-range-output", "children"),
    Input("af-launch-mgrs", "value"),
    Input("af-impact-mgrs", "value"),
)
def display_range(launch_mgrs, impact_mgrs):
    if not launch_mgrs or not impact_mgrs:
        return ""
    try:
        launch = zf.MGRS2LL(launch_mgrs)
        impact = zf.MGRS2LL(impact_mgrs)
        distance, _ = zf.LLDist(launch[1], launch[2], impact[1], impact[2])
        return f"Range: {distance:,.0f} m"
    except Exception as exc:
        return f"Range unavailable: {exc}"


@callback(
    Output("af-velocity", "value"),
    Input("af-launch-mgrs", "value"),
    Input("af-impact-mgrs", "value"),
    Input("af-mass", "value"),
)
def suggest_initial_velocity(launch_mgrs, impact_mgrs, mass):
    if not launch_mgrs or not impact_mgrs or mass is None:
        return None
    try:
        launch = zf.MGRS2LL(launch_mgrs)
        impact = zf.MGRS2LL(impact_mgrs)
        distance, _ = zf.LLDist(launch[1], launch[2], impact[1], impact[2])
        mass = float(mass)
        if distance <= 12000:
            velocity = -2.305882 * mass + 806.6588
        elif distance <= 18000:
            velocity = -2.105882 * mass + 899.9588
        else:
            velocity = -2.447059 * mass + 1043.271
        return round(velocity, 1)
    except Exception:
        return None


def _make_map(launch, impact, azimuth_of_lay):
    launch_lat, launch_lon = float(launch[1]), float(launch[2])
    impact_lat, impact_lon = float(impact[1]), float(impact[2])
    lon_delta = ((impact_lon - launch_lon + 180) % 360) - 180
    center_lon = ((launch_lon + lon_delta / 2 + 180) % 360) - 180
    distance, _ = zf.LLDist(launch_lat, launch_lon, impact_lat, impact_lon)
    zoom = max(2, min(15, int(-1.36 * math.log(max(distance, 1) / 1000) + 15)))
    fmap = folium.Map(location=[(launch_lat + impact_lat) / 2, center_lon], zoom_start=zoom)
    folium.TileLayer("OpenStreetMap", name="Open Street Map").add_to(fmap)
    for layer, name in (("m", "Google Maps"), ("s", "Google Satellite"), ("p", "Google Terrain")):
        folium.TileLayer(
            tiles=f"https://mt1.google.com/vt/lyrs={layer}&x={{x}}&y={{y}}&z={{z}}",
            attr="Google",
            name=name,
            overlay=False,
            control=True,
        ).add_to(fmap)
    folium.LayerControl().add_to(fmap)
    fmap.add_child(plugins.MiniMap(toggle_display=True))
    plugins.ScrollZoomToggler().add_to(fmap)
    plugins.Fullscreen(position="topright").add_to(fmap)

    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    icons_dir = os.path.join(app_dir, "Icons")
    launch_icon_path = os.path.join(icons_dir, "paladin.jpg")
    impact_icon_path = os.path.join(icons_dir, "target.png")
    launch_icon = folium.features.CustomIcon(launch_icon_path, icon_size=(30, 20)) if os.path.exists(launch_icon_path) else folium.Icon(color="green")
    impact_icon = folium.features.CustomIcon(impact_icon_path, icon_size=(25, 25)) if os.path.exists(impact_icon_path) else folium.Icon(color="red")
    folium.Marker([launch_lat, launch_lon], tooltip="Launch Point", icon=launch_icon).add_to(fmap)
    folium.Marker([impact_lat, impact_lon], tooltip="Impact Point", icon=impact_icon).add_to(fmap)
    folium.PolyLine([[launch_lat, launch_lon], [impact_lat, impact_lon]], color="red", weight=3).add_to(fmap)
    folium.Circle([launch_lat, launch_lon], radius=20000, color="red", fill=True, fill_color="red", fill_opacity=0.1).add_to(fmap)
    lay_endpoint = zf.vPolar(launch_lat, launch_lon, float(azimuth_of_lay) * 180 / 3200, 20000)
    folium.PolyLine([[launch_lat, launch_lon], list(lay_endpoint)], color="blue", weight=3, tooltip="Azimuth of lay").add_to(fmap)
    fmap.add_child(folium.LatLngPopup())
    return fmap._repr_html_()


def _missing_solver_message():
    expected = ("MLAHshot", "MLALshot", "projectile_motion_with_drag")
    missing = [name for name in expected if not callable(getattr(zf, name, None))]
    if not missing:
        return None
    return (
        "The map and range are available, but the ballistic solution and trajectory "
        "cannot be calculated because these functions are missing from "
        f"SubApps/z_functions.py: {', '.join(missing)}."
    )


@callback(
    Output("af-map-iframe", "srcDoc"),
    Output("af-solution-output", "children"),
    Output("af-trajectory-graph", "figure"),
    Input("af-launch-mgrs", "value"),
    Input("af-impact-mgrs", "value"),
    Input("af-launch-altitude", "value"),
    Input("af-impact-altitude", "value"),
    Input("af-mass", "value"),
    Input("af-velocity", "value"),
    Input("af-aol", "value"),
    Input("af-angle-mode", "value"),
)
def update_shot(launch_mgrs, impact_mgrs, launch_alt, impact_alt, mass, velocity, aol, angle_mode):
    empty_figure = go.Figure()
    if not launch_mgrs or not impact_mgrs or mass is None or velocity is None or aol is None:
        return "", "Enter valid coordinates and projectile parameters.", empty_figure

    try:
        launch = zf.MGRS2LL(launch_mgrs)
        impact = zf.MGRS2LL(impact_mgrs)
        launch_lat, launch_lon = float(launch[1]), float(launch[2])
        impact_lat, impact_lon = float(impact[1]), float(impact[2])
        range_m, azimuth_deg = zf.LLDist(launch_lat, launch_lon, impact_lat, impact_lon)
        map_html = _make_map(launch, impact, aol)
        launch_mils = azimuth_deg * 3200 / 180
        diagnostics = html.Div(
            [
                html.P(f"Range: {range_m:,.0f} m", className="mb-1"),
                html.P(f"Launch Bearing: {azimuth_deg:.2f}°", className="mb-1"),
                html.P(f"Launch Azimuth: {launch_mils:.2f} mils", className="mb-1"),
            ],
            className="mb-2",
        )

        missing_message = _missing_solver_message()
        if missing_message:
            return map_html, html.Div([diagnostics, dbc.Alert(missing_message, color="warning")]), empty_figure

        high_angle = angle_mode == "high"
        solver = zf.MLAHshot if high_angle else zf.MLALshot
        solution = solver(
            launch_lat,
            float(launch_alt),
            float(impact_alt),
            float(mass),
            launch_mils,
            float(aol),
            float(velocity),
            range_m,
        )
        mass_kg = float(mass) / 2.2
        diameter_mm = 155
        area = math.pi * (diameter_mm / 2000) ** 2
        spin_rate = float(velocity) / (20 * diameter_mm / 1000)
        spin_velocity = spin_rate * 2 * math.pi
        elevation_deg = float(solution[0]) / 3200 * 180
        motion = zf.projectile_motion_with_drag(
            float(velocity), elevation_deg, mass_kg, 0.25, area,
            launch_lat, azimuth_deg, float(launch_alt), float(launch_alt), spin_velocity,
        )
        time_of_flight, final_x, final_y, final_z, times, x_vals, y_vals, z_vals, vx, vy, vz, v_mag, qe = motion
        import pandas as pd

        trajectory = pd.DataFrame({"time": times, "distance": x_vals, "height": y_vals})
        trajectory["distance"] = trajectory["distance"] / max(trajectory["distance"]) * range_m
        trajectory["height"] = (
            trajectory["height"] - float(launch_alt)
        ) / max(trajectory["height"] - float(launch_alt)) * (float(solution[1]) - float(launch_alt)) + float(launch_alt)
        figure = px.line(
            trajectory,
            x="distance",
            y="height",
            title="Projectile trajectory",
            labels={"distance": "Distance (m)", "height": "Height (m)"},
        )
        figure.update_layout(height=350, margin={"l": 30, "r": 20, "t": 45, "b": 35})

        output = html.Div(
            [
                diagnostics,
                html.P(f"Max Ord: {int(solution[1])} m", className="mb-1"),
                html.P(f"Time of Flight: {int(solution[2])} s", className="mb-1"),
                html.P(f"QE: {int(solution[0])} mils", className="mb-1"),
                html.P(f"Deflection: {int(solution[4])} mils", className="mb-1"),
                html.P(f"Drift: {int(solution[3])} mils", className="mb-1"),
                html.P(f"Final Velocity: {int(v_mag[-10])} m/s", className="mb-1"),
                html.P(f"Final QE: {int(qe[-10])} mils", className="mb-0"),
            ]
        )
        return map_html, output, figure
    except Exception as exc:
        return "", dbc.Alert(f"Could not calculate shot: {exc}", color="danger", className="mb-0"), empty_figure
