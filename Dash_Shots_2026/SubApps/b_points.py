"""Point-to-point MGRS planning page for the Dash Shots app."""

import math
import os

import dash
from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
import folium
from folium import plugins

from SubApps import z_functions as zf


dash.register_page(__name__, path="/b_points", name="Point to Point", order=2)


layout = dbc.Container(
    [
        html.H2("Point to Point", className="mb-4"),
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
                                                    id="bp-lookup",
                                                    type="text",
                                                    placeholder="Place or address (3+ characters)",
                                                ),
                                                dbc.Button("Lookup", id="bp-lookup-button", n_clicks=0),
                                            ],
                                            className="mb-2",
                                        ),
                                        html.Div(id="bp-lookup-output"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Point coordinates"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Launch Point (MGRS)"),
                                        dbc.Input(
                                            id="bp-launch-mgrs",
                                            type="text",
                                            value="12RWU1059022575",
                                            className="mb-2",
                                        ),
                                        html.Div(id="bp-launch-ll", className="text-muted mb-3"),
                                        dbc.Label("Impact Point (MGRS)"),
                                        dbc.Input(
                                            id="bp-impact-mgrs",
                                            type="text",
                                            value="31UDQ5248411718",
                                            className="mb-2",
                                        ),
                                        html.Div(id="bp-impact-ll", className="text-muted"),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Point-to-point details"),
                                dbc.CardBody(html.Div(id="bp-details-output")),
                            ]
                        ),
                    ],
                    xs=12,
                    lg=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Route map"),
                            dbc.CardBody(
                                html.Iframe(
                                    id="bp-map-iframe",
                                    title="Map showing the launch to impact route",
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
    Output("bp-lookup-output", "children"),
    Input("bp-lookup-button", "n_clicks"),
    State("bp-lookup", "value"),
    prevent_initial_call=True,
)
def process_lookup(n_clicks, query):
    if not n_clicks or not query or len(query.strip()) < 3:
        return "Enter at least three characters to search."
    try:
        place = zf.lookup(query)
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
    Output("bp-launch-ll", "children"),
    Input("bp-launch-mgrs", "value"),
)
def display_launch_coordinates(mgrs):
    if not mgrs:
        return ""
    try:
        point = zf.MGRS2LL(mgrs)
        return f"Launch Point (LL): {point[1]:.5f}, {point[2]:.5f}"
    except Exception as exc:
        return f"Invalid MGRS: {exc}"


@callback(
    Output("bp-impact-ll", "children"),
    Input("bp-impact-mgrs", "value"),
)
def display_impact_coordinates(mgrs):
    if not mgrs:
        return ""
    try:
        point = zf.MGRS2LL(mgrs)
        return f"Impact Point (LL): {point[1]:.5f}, {point[2]:.5f}"
    except Exception as exc:
        return f"Invalid MGRS: {exc}"


def _route_points(start, bearing, distance, end, segments=50):
    """Build a smooth geodesic route using evenly spaced samples."""
    if distance == 0:
        return [[start[1], start[2]], [end[1], end[2]]]
    points = []
    for index in range(segments + 1):
        lat, lon = zf.vPolar(start[1], start[2], bearing, distance * index / segments)
        points.append([lat, lon])
    points[-1] = [end[1], end[2]]
    return points


@callback(
    Output("bp-map-iframe", "srcDoc"),
    Output("bp-details-output", "children"),
    Input("bp-launch-mgrs", "value"),
    Input("bp-impact-mgrs", "value"),
)
def update_map_and_details(launch_mgrs, impact_mgrs):
    if not launch_mgrs or not impact_mgrs:
        return "", "Enter both MGRS coordinates."
    try:
        launch = zf.MGRS2LL(launch_mgrs)
        impact = zf.MGRS2LL(impact_mgrs)
        distance, launch_bearing = zf.LLDist(launch[1], launch[2], impact[1], impact[2])
        _, impact_bearing = zf.LLDist(impact[1], impact[2], launch[1], launch[2])
        launch_mils = launch_bearing * 3200 / 180
        impact_mils = impact_bearing * 3200 / 180

        # Keep the center correct for routes crossing the antimeridian.
        lon_delta = ((impact[2] - launch[2] + 180) % 360) - 180
        center_lon = ((launch[2] + lon_delta / 2 + 180) % 360) - 180
        zoom = max(2, min(14, int(-1.36 * math.log(max(distance, 1) / 1000) + 14)))
        fmap = folium.Map(location=[(launch[1] + impact[1]) / 2, center_lon], zoom_start=zoom)

        folium.TileLayer("OpenStreetMap", name="Open Street Map").add_to(fmap)
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            attr="Google",
            name="Google Maps",
            overlay=False,
            control=True,
        ).add_to(fmap)
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            name="Google Satellite",
            overlay=False,
            control=True,
        ).add_to(fmap)
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
            attr="Google",
            name="Google Terrain",
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
            [launch[1], launch[2]], popup=launch_mgrs, tooltip="Launch Point", icon=launch_icon
        ).add_to(fmap)
        folium.Marker(
            [impact[1], impact[2]], popup=impact_mgrs, tooltip="Impact Point", icon=impact_icon
        ).add_to(fmap)
        folium.PolyLine(
            _route_points(launch, launch_bearing, distance, impact),
            color="red",
            tooltip="Point-to-point route",
        ).add_to(fmap)
        plugins.Draw().add_to(fmap)

        details = dbc.Row(
            [
                dbc.Col([html.Strong("Distance: "), f"{round(distance):,} meters"], xs=12, md=6),
                dbc.Col([html.Strong("Launch Bearing: "), f"{launch_bearing:.2f}°"], xs=12, md=6),
                dbc.Col([html.Strong("Launch Azimuth: "), f"{launch_mils:.2f} mils"], xs=12, md=6),
                dbc.Col([html.Strong("Impact Bearing: "), f"{impact_bearing:.2f}°"], xs=12, md=6),
                dbc.Col([html.Strong("Impact Azimuth: "), f"{impact_mils:.2f} mils"], xs=12, md=6),
            ],
            className="g-2",
        )
        return fmap._repr_html_(), details
    except Exception as exc:
        return "", dbc.Alert(f"Could not calculate route: {exc}", color="danger", className="mb-0")
