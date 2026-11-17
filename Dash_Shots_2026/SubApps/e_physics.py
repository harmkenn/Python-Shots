"""Physics-based projectile planning page for the Dash Shots app."""

import math
import os

import dash
from dash import Input, Output, State, callback, dcc, html
import dash_bootstrap_components as dbc
import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from folium import plugins

from SubApps import z_functions as zf


dash.register_page(__name__, path="/e_physics", name="Physics Shot", order=5)


layout = dbc.Container(
    [
        html.H2("Physics Shot", className="mb-4"),
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
                                                    id="ep-lookup",
                                                    type="text",
                                                    placeholder="Place or address (3+ characters)",
                                                ),
                                                dbc.Button("Lookup", id="ep-lookup-button", n_clicks=0),
                                            ],
                                            className="mb-2",
                                        ),
                                        html.Div(id="ep-lookup-output"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Launch point"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Launch Point (MGRS)"),
                                        dbc.Input(
                                            id="ep-launch-mgrs",
                                            type="text",
                                            value="36SYC6482056356",
                                            className="mb-2",
                                        ),
                                        dbc.Label("Launch Altitude (meters)"),
                                        dbc.Input(id="ep-launch-altitude", type="number", value=150, step="any"),
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
                                        dbc.Label("Initial Velocity (m/s)"),
                                        dbc.Input(id="ep-velocity", type="number", value=810.8, min=0, step="any", className="mb-2"),
                                        dbc.Label("Launch Angle (mils)"),
                                        dbc.Input(id="ep-launch-angle", type="number", value=357.4, step="any", className="mb-2"),
                                        dbc.Label("Mass (lbs)"),
                                        dbc.Input(id="ep-mass", type="number", value=95, min=0, step="any", className="mb-2"),
                                        dbc.Label("Diameter (mm)"),
                                        dbc.Input(id="ep-diameter", type="number", value=155, min=0, step="any", className="mb-2"),
                                        dbc.Label("Gun-Target Azimuth (mils)"),
                                        dbc.Input(id="ep-gt-azimuth", type="number", value=5731, step="any", className="mb-2"),
                                        dbc.Label("Azimuth of Lay (mils)"),
                                        dbc.Input(id="ep-aol", type="number", value=1234, step="any"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Physics and model results"),
                                dbc.CardBody(html.Div(id="ep-results-output")),
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
                                dbc.CardHeader("Impact map"),
                                dbc.CardBody(
                                    dcc.Loading(
                                        html.Iframe(
                                            id="ep-map-iframe",
                                            title="Map showing the launch and predicted impact points",
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
                                dbc.CardBody(dcc.Graph(id="ep-trajectory-graph", figure=go.Figure())),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Trajectory samples"),
                                dbc.CardBody(html.Div(id="ep-trajectory-table", className="table-responsive")),
                            ]
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
    Output("ep-lookup-output", "children"),
    Input("ep-lookup-button", "n_clicks"),
    State("ep-lookup", "value"),
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
    Output("ep-launch-altitude", "value"),
    Input("ep-launch-mgrs", "value"),
)
def set_launch_altitude(mgrs):
    if not mgrs:
        return None
    try:
        launch = zf.MGRS2LL(mgrs)
        return round(float(zf.elevation(launch[1], launch[2])), 2)
    except Exception:
        return None


def _trajectory_figure(data, model_range):
    figure = px.line(
        data,
        x="x2",
        y="y2",
        title="Drag, varying air density and gravity",
        labels={"x2": "Distance (m)", "y2": "Height (m)"},
    )
    figure.update_layout(height=380, margin={"l": 35, "r": 20, "t": 55, "b": 40})
    figure.update_xaxes(range=[0, max(float(model_range), 1)])
    figure.update_yaxes(rangemode="tozero", scaleanchor="x", scaleratio=1)
    return figure


def _build_map(launch, impact, launch_mgrs, impact_mgrs, bearing, distance):
    launch_lat, launch_lon = float(launch[1]), float(launch[2])
    impact_lat, impact_lon = float(impact[0]), float(impact[1])
    lon_delta = ((impact_lon - launch_lon + 180) % 360) - 180
    center_lon = ((launch_lon + lon_delta / 2 + 180) % 360) - 180
    zoom = max(2, min(15, int(-1.36 * math.log(max(float(distance), 1) / 1000) + 15)))
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
    launch_icon = (
        folium.features.CustomIcon(launch_icon_path, icon_size=(30, 20))
        if os.path.exists(launch_icon_path)
        else folium.Icon(color="green")
    )
    impact_icon = (
        folium.features.CustomIcon(impact_icon_path, icon_size=(25, 25))
        if os.path.exists(impact_icon_path)
        else folium.Icon(color="red")
    )
    folium.Marker(
        [launch_lat, launch_lon], popup=launch_mgrs, tooltip="Launch Point", icon=launch_icon
    ).add_to(fmap)
    folium.Marker(
        [impact_lat, impact_lon], popup=impact_mgrs, tooltip="Impact Point", icon=impact_icon
    ).add_to(fmap)
    folium.PolyLine(
        [[launch_lat, launch_lon], [impact_lat, impact_lon]],
        color="red",
        weight=3,
        tooltip="Predicted trajectory ground track",
    ).add_to(fmap)
    plugins.Draw().add_to(fmap)
    fmap.add_child(folium.LatLngPopup())
    return fmap._repr_html_()


@callback(
    Output("ep-map-iframe", "srcDoc"),
    Output("ep-results-output", "children"),
    Output("ep-trajectory-graph", "figure"),
    Output("ep-trajectory-table", "children"),
    Input("ep-launch-mgrs", "value"),
    Input("ep-launch-altitude", "value"),
    Input("ep-velocity", "value"),
    Input("ep-launch-angle", "value"),
    Input("ep-mass", "value"),
    Input("ep-diameter", "value"),
    Input("ep-gt-azimuth", "value"),
    Input("ep-aol", "value"),
)
def update_physics_shot(launch_mgrs, launch_altitude, velocity, launch_angle_mils, mass_lbs, diameter_mm, gt_azimuth_mils, aol_mils):
    empty_figure = go.Figure()
    if any(value is None for value in (launch_altitude, velocity, launch_angle_mils, mass_lbs, diameter_mm, gt_azimuth_mils, aol_mils)):
        return "", "Enter all projectile and launch parameters.", empty_figure, ""

    try:
        launch = zf.MGRS2LL(launch_mgrs)
        launch_lat, launch_lon = float(launch[1]), float(launch[2])
        launch_altitude = float(launch_altitude)
        velocity = float(velocity)
        launch_angle_mils = float(launch_angle_mils)
        mass_lbs = float(mass_lbs)
        diameter_mm = float(diameter_mm)
        gt_azimuth_mils = float(gt_azimuth_mils)
        aol_mils = float(aol_mils)
        if velocity <= 0 or mass_lbs <= 0 or diameter_mm <= 0:
            raise ValueError("Velocity, mass, and diameter must be greater than zero.")

        launch_angle_deg = launch_angle_mils * 180 / 3200
        gt_azimuth_deg = gt_azimuth_mils * 180 / 3200
        mass_kg = mass_lbs / 2.2
        area = math.pi * (diameter_mm / 2000) ** 2
        spin_rate = velocity / (20 * diameter_mm / 1000)
        spin_velocity = spin_rate * 2 * math.pi

        motion = zf.projectile_motion_with_drag(
            velocity,
            launch_angle_deg,
            mass_kg,
            0.25,
            area,
            launch_lat,
            gt_azimuth_deg,
            launch_altitude,
            launch_altitude,
            spin_velocity,
        )
        time_of_flight, final_x, final_y, final_z, times, x_values, y_values, z_values, vx, vy, vz, speed, qe = motion

        model_range, max_ordinate, model_tof, model_drift, deflection = zf.MLshot(
            launch_lat,
            launch_altitude,
            launch_altitude,
            mass_lbs,
            gt_azimuth_mils,
            aol_mils,
            launch_angle_mils,
            velocity,
        )

        data = pd.DataFrame(
            {
                "t": np.asarray(times, dtype=float),
                "x": np.asarray(x_values, dtype=float),
                "y": np.asarray(y_values, dtype=float),
                "z": np.asarray(z_values, dtype=float),
                "vx": np.asarray(vx, dtype=float),
                "vy": np.asarray(vy, dtype=float),
                "vz": np.asarray(vz, dtype=float),
                "QE": np.asarray(qe, dtype=float),
                "v_mag": np.asarray(speed, dtype=float),
            }
        )
        if len(data) < 2:
            raise ValueError("The projectile model returned too few trajectory points.")
        if data["t"].max() <= 0 or data["x"].max() <= 0:
            raise ValueError("The projectile model did not produce a valid flight path.")

        model_range = float(model_range)
        max_ordinate = float(max_ordinate)
        model_tof = float(model_tof)
        model_drift = float(model_drift)
        data["t2"] = data["t"] / data["t"].max() * model_tof
        data["x2"] = data["x"] / data["x"].max() * model_range
        height_gain = data["y"] - launch_altitude
        max_height_gain = float(height_gain.max())
        if max_height_gain > 0:
            data["y2"] = height_gain / max_height_gain * (max_ordinate - launch_altitude) + launch_altitude
        else:
            data["y2"] = data["y"]

        data["drift_calc"] = np.arctan2(data["z"], data["x"]) / math.pi * 3200
        drift_window = data["drift_calc"].iloc[-5:]
        window_max = float(drift_window.max())
        if abs(window_max) > 1e-9:
            data["drift_mils"] = data["drift_calc"] / window_max * model_drift
        else:
            data["drift_mils"] = np.linspace(0, model_drift, len(data))
        data["z2"] = data["x2"] * np.tan(data["drift_mils"] / 3200 * math.pi)
        data["dx"] = data["x2"].diff()
        data["dy"] = data["y2"].diff()
        data["dt"] = data["t2"].diff()
        data["qe2"] = np.arctan2(data["dy"], data["dx"]) / math.pi * 3200
        data["v2"] = np.hypot(data["dy"], data["dx"]) / data["dt"].replace(0, np.nan)
        sample_index = max(0, len(data) - 10)
        final_velocity = data["v2"].iloc[sample_index]
        final_qe = data["qe2"].iloc[sample_index]
        if pd.isna(final_velocity):
            final_velocity = velocity
        if pd.isna(final_qe):
            final_qe = launch_angle_mils

        impact_bearing = (gt_azimuth_deg + model_drift / 3200 * 180) % 360
        impact_lat, impact_lon = zf.vPolar(launch_lat, launch_lon, impact_bearing, float(final_x))
        impact = (impact_lat, impact_lon)
        impact_mgrs = zf.LL2MGRS(impact_lat, impact_lon)[1]
        map_html = _build_map(
            launch,
            impact,
            launch_mgrs,
            impact_mgrs,
            impact_bearing,
            float(final_x),
        )

        figure = _trajectory_figure(data, model_range)
        results = dbc.Row(
            [
                dbc.Col([html.Strong("ML range: "), f"{model_range:,.0f} m"], xs=12, md=6),
                dbc.Col([html.Strong("Max ordinate: "), f"{max_ordinate:,.0f} m"], xs=12, md=6),
                dbc.Col([html.Strong("Time of flight: "), f"{model_tof:.1f} s"], xs=12, md=6),
                dbc.Col([html.Strong("Launch angle: "), f"{launch_angle_deg:.2f}° ({launch_angle_mils:.2f} mils)"], xs=12, md=6),
                dbc.Col([html.Strong("Drift: "), f"{model_drift:.2f} mils"], xs=12, md=6),
                dbc.Col([html.Strong("Deflection: "), f"{float(deflection):.2f} mils"], xs=12, md=6),
                dbc.Col([html.Strong("Drift distance: "), f"{data['z2'].iloc[sample_index]:.2f} m"], xs=12, md=6),
                dbc.Col([html.Strong("Final velocity: "), f"{float(final_velocity):.2f} m/s"], xs=12, md=6),
                dbc.Col([html.Strong("Final QE: "), f"{float(final_qe):.2f} mils"], xs=12, md=6),
                dbc.Col([html.Strong("Launch point (LL): "), f"{launch_lat:.5f}, {launch_lon:.5f}, {launch_altitude:.0f} m"], xs=12),
                dbc.Col([html.Strong("Impact location (MGRS): "), impact_mgrs], xs=12),
                dbc.Col([html.Strong("Impact point (LL): "), f"{impact_lat:.5f}, {impact_lon:.5f}"], xs=12),
            ],
            className="g-2",
        )
        table_columns = [
            ("Time (s)", "t2"),
            ("Distance (m)", "x2"),
            ("Height (m)", "y2"),
            ("Drift (m)", "z2"),
            ("Velocity (m/s)", "v2"),
            ("QE (mils)", "qe2"),
        ]
        table = dbc.Table(
            [
                html.Thead(html.Tr([html.Th(label) for label, _ in table_columns])),
                html.Tbody(
                    [
                        html.Tr(
                            [
                                html.Td(f"{float(data[column].iloc[index]):.2f}" if pd.notna(data[column].iloc[index]) else "—")
                                for _, column in table_columns
                            ]
                        )
                        for index in np.linspace(0, len(data) - 1, min(12, len(data)), dtype=int)
                    ]
                ),
            ],
            bordered=True,
            hover=True,
            striped=True,
            size="sm",
        )
        return map_html, results, figure, table
    except Exception as exc:
        return "", dbc.Alert(f"Could not calculate physics shot: {exc}", color="danger", className="mb-0"), empty_figure, ""
