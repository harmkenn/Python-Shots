"""Polar shot planning page for the Dash Shots app."""

import math
import os

import dash
from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
import folium
from folium import plugins

from SubApps import z_functions as zf


dash.register_page(__name__, path="/c_polar", name="Polar Shot", order=3)


layout = dbc.Container(
    [
        html.H2("Polar Shot", className="mb-4"),
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
                                                    id="cp-lookup",
                                                    type="text",
                                                    placeholder="Place or address (3+ characters)",
                                                ),
                                                dbc.Button("Lookup", id="cp-lookup-button", n_clicks=0),
                                            ],
                                            className="mb-2",
                                        ),
                                        html.Div(id="cp-lookup-output"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Launch parameters"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Launch Point (MGRS)"),
                                        dbc.Input(
                                            id="cp-launch-mgrs",
                                            type="text",
                                            value="12RWU1059022575",
                                            className="mb-2",
                                        ),
                                        html.Div(id="cp-launch-ll", className="text-muted mb-3"),
                                        dbc.Label("Azimuth (mils)"),
                                        dbc.Input(
                                            id="cp-azimuth",
                                            type="number",
                                            value=1600,
                                            step="any",
                                            className="mb-2",
                                        ),
                                        dbc.Label("Distance (meters)"),
                                        dbc.Input(
                                            id="cp-distance",
                                            type="number",
                                            value=10000000,
                                            min=0,
                                            step="any",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Polar shot details"),
                                dbc.CardBody(html.Div(id="cp-details-output")),
                            ]
                        ),
                    ],
                    xs=12,
                    lg=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Trajectory map"),
                            dbc.CardBody(
                                html.Iframe(
                                    id="cp-map-iframe",
                                    title="Map showing the polar shot trajectory",
                                    style={"width": "100%", "height": "650px", "border": "none"},
                                )
                            ),
                        ]
                    ),
                    xs=12,
                    lg=8,
                ),
            ],
            className="g-3",
        ),
    ],
    fluid=True,
)


@callback(
    Output("cp-lookup-output", "children"),
    Input("cp-lookup-button", "n_clicks"),
    State("cp-lookup", "value"),
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
    Output("cp-launch-ll", "children"),
    Input("cp-launch-mgrs", "value"),
)
def display_launch_coordinates(mgrs):
    if not mgrs:
        return ""
    try:
        point = zf.MGRS2LL(mgrs)
        return f"Launch Point (LL): {point[1]:.5f}, {point[2]:.5f}"
    except Exception as exc:
        return f"Invalid MGRS: {exc}"


def _trajectory_points(launch, bearing, distance, impact, segments=100):
    """Sample the geodesic from launch to impact for a smooth map line."""
    if distance == 0:
        return [[launch[1], launch[2]], [impact[0], impact[1]]]

    points = []
    for index in range(segments + 1):
        lat, lon = zf.vPolar(launch[1], launch[2], bearing, distance * index / segments)
        points.append([lat, lon])
    points[-1] = [impact[0], impact[1]]
    return points


@callback(
    Output("cp-map-iframe", "srcDoc"),
    Output("cp-details-output", "children"),
    Input("cp-launch-mgrs", "value"),
    Input("cp-azimuth", "value"),
    Input("cp-distance", "value"),
)
def update_map_and_details(launch_mgrs, azimuth_mils, distance_meters):
    if not launch_mgrs or azimuth_mils is None or distance_meters is None:
        return "", "Enter a launch point, azimuth, and distance."

    try:
        azimuth_mils = float(azimuth_mils)
        distance_meters = float(distance_meters)
        if distance_meters < 0:
            raise ValueError("Distance must be zero or greater.")

        launch = zf.MGRS2LL(launch_mgrs)
        launch_lat, launch_lon = float(launch[1]), float(launch[2])
        launch_bearing = azimuth_mils * 180 / 3200
        impact_lat, impact_lon = zf.vPolar(
            launch_lat, launch_lon, launch_bearing, distance_meters
        )
        _, impact_bearing = zf.LLDist(impact_lat, impact_lon, launch_lat, launch_lon)
        impact = zf.LL2MGRS(impact_lat, impact_lon)
        impact_mgrs = impact[1]
        impact_mils = impact_bearing * 3200 / 180

        # Center across the antimeridian using the shorter longitude interval.
        lon_delta = ((impact_lon - launch_lon + 180) % 360) - 180
        center_lon = ((launch_lon + lon_delta / 2 + 180) % 360) - 180
        zoom = max(2, min(15, int(-1.36 * math.log(max(distance_meters, 1) / 1000) + 15)))
        fmap = folium.Map(
            location=[(launch_lat + impact_lat) / 2, center_lon],
            zoom_start=zoom,
        )

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
            [launch_lat, launch_lon],
            popup=launch_mgrs,
            tooltip="Launch Point",
            icon=launch_icon,
        ).add_to(fmap)
        folium.Marker(
            [impact_lat, impact_lon],
            popup=impact_mgrs,
            tooltip="Impact Point",
            icon=impact_icon,
        ).add_to(fmap)
        folium.PolyLine(
            _trajectory_points(
                launch,
                launch_bearing,
                distance_meters,
                (impact_lat, impact_lon),
            ),
            color="red",
            tooltip="Polar shot trajectory",
        ).add_to(fmap)
        plugins.Draw().add_to(fmap)

        details = dbc.Row(
            [
                dbc.Col([html.Strong("Distance: "), f"{distance_meters:,.2f} meters"], xs=12, md=6),
                dbc.Col([html.Strong("Launch Bearing: "), f"{launch_bearing:.2f}°"], xs=12, md=6),
                dbc.Col([html.Strong("Launch Azimuth: "), f"{azimuth_mils:.2f} mils"], xs=12, md=6),
                dbc.Col([html.Strong("Impact Bearing: "), f"{impact_bearing:.2f}°"], xs=12, md=6),
                dbc.Col([html.Strong("Impact Azimuth: "), f"{impact_mils:.2f} mils"], xs=12, md=6),
                dbc.Col([html.Strong("Impact Location (MGRS): "), impact_mgrs], xs=12),
                dbc.Col(
                    [html.Strong("Impact Point (LL): "), f"{impact_lat:.5f}, {impact_lon:.5f}"],
                    xs=12,
                ),
            ],
            className="g-2",
        )
        return fmap._repr_html_(), details
    except Exception as exc:
        return "", dbc.Alert(f"Could not calculate polar shot: {exc}", color="danger", className="mb-0")
