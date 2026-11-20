"""Find the Sun's direction from an observer location and UTC time."""

import math
from datetime import datetime, timezone

import dash
from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
from SubApps.maplibre_utils import render_geojson_map
from SubApps import z_functions as zf


dash.register_page(__name__, path="/g_sunnorth", name="Sun North", order=7)


layout = dbc.Container(
    [
        html.H2("Get Your Position from the Location of the Sun", className="mb-4"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("UTC date and time"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Date (UTC)"),
                                        dbc.Input(id="sn-date", type="date", value="2022-04-09", className="mb-2"),
                                        dbc.Row(
                                            [
                                                dbc.Col([dbc.Label("Hour"), dbc.Input(id="sn-hour", type="number", min=0, max=23, value=22)], width=4),
                                                dbc.Col([dbc.Label("Minute"), dbc.Input(id="sn-minute", type="number", min=0, max=59, value=40)], width=4),
                                                dbc.Col([dbc.Label("Second"), dbc.Input(id="sn-second", type="number", min=0, max=59, value=0)], width=4),
                                            ], className="mb-2",
                                        ),
                                        dbc.Button("Use current UTC time", id="sn-now-button", n_clicks=0, color="secondary", outline=True),
                                    ]
                                ),
                            ], className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Your location"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Location (MGRS)"),
                                        dbc.Input(id="sn-mgrs", type="text", value="32TPR9792346676", className="mb-2"),
                                        html.Div(id="sn-location-output"),
                                    ]
                                ),
                            ], className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Location lookup"),
                                dbc.CardBody(
                                    [
                                        dbc.InputGroup(
                                            [
                                                dbc.Input(id="sn-lookup", type="text", placeholder="Place or address (3+ characters)"),
                                                dbc.Button("Lookup", id="sn-lookup-button", n_clicks=0),
                                            ], className="mb-2",
                                        ),
                                        html.Div(id="sn-lookup-output"),
                                    ]
                                ),
                            ], className="mb-3",
                        ),
                        dbc.Card([dbc.CardHeader("Sun position"), dbc.CardBody(html.Div(id="sn-results-output"))]),
                    ], xs=12, lg=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Subsolar point and observer location"),
                            dbc.CardBody(
                                html.Iframe(
                                    id="sn-map-iframe",
                                    title="Map showing the subsolar point, observer, and direction to the Sun",
                                    style={"width": "100%", "height": "700px", "border": "none"},
                                )
                            ),
                        ]
                    ), xs=12, lg=8,
                ),
            ], className="g-3",
        ),
    ], fluid=True,
)


@callback(
    Output("sn-date", "value"), Output("sn-hour", "value"),
    Output("sn-minute", "value"), Output("sn-second", "value"),
    Input("sn-now-button", "n_clicks"), prevent_initial_call=True,
)
def set_current_utc_time(n_clicks):
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d"), now.hour, now.minute, now.second


@callback(
    Output("sn-lookup-output", "children"), Output("sn-mgrs", "value"),
    Input("sn-lookup-button", "n_clicks"), State("sn-lookup", "value"),
    prevent_initial_call=True,
)
def process_lookup(n_clicks, query):
    if not n_clicks or not query or len(query.strip()) < 3:
        return "Enter at least three characters to search.", dash.no_update
    try:
        place = zf.lookup(query.strip())
        altitude = zf.elevation(place[1], place[2])
        mgrs = place[3]
        return html.Div(
            [
                html.P(place[0], className="mb-1 fw-bold"),
                html.P(f"Lat: {place[1]} | Lon: {place[2]}", className="mb-1"),
                html.P(f"MGRS: {mgrs}", className="mb-1"),
                html.P(f"Alt: {round(altitude, 2)} meters", className="mb-2"),
            ]
        ), mgrs
    except Exception as exc:
        return dbc.Alert(f"Lookup error: {exc}", color="danger", className="mb-0"), dash.no_update


def _route_points(observer, subsolar, bearing, distance_m, segments=100):
    if distance_m <= 1e-6:
        return [list(observer), list(subsolar)]
    points = [
        list(zf.vPolar(observer[0], observer[1], bearing, distance_m * i / segments))
        for i in range(segments + 1)
    ]
    points[-1] = list(subsolar)
    return points


def _map_html(subsolar, observer, bearing, distance_m):
    sun_lat, sun_lon = subsolar
    obs_lat, obs_lon = observer
    lon_delta = ((obs_lon - sun_lon + 180) % 360) - 180
    center_lon = ((sun_lon + lon_delta / 2 + 180) % 360) - 180
    center_lat = (sun_lat + obs_lat) / 2
    zoom = max(2, min(15, int(-1.36 * math.log(max(distance_m / 1000, 1)) + 14)))
    sun_mgrs = zf.LL2MGRS(sun_lat, sun_lon)[1]
    observer_mgrs = zf.LL2MGRS(obs_lat, obs_lon)[1]
    features = [
        {"type": "Feature", "properties": {"kind": "subsolar", "label": "Subsolar Point", "popup": f"Subsolar Point MGRS: {sun_mgrs}", "color": "#f59e0b", "radius": 9}, "geometry": {"type": "Point", "coordinates": [sun_lon, sun_lat]}},
        {"type": "Feature", "properties": {"kind": "observer", "label": "Observer", "popup": f"Observer MGRS: {observer_mgrs}", "color": "#1976d2", "radius": 8}, "geometry": {"type": "Point", "coordinates": [obs_lon, obs_lat]}},
    ]
    route = [[float(lon), float(lat)] for lat, lon in _route_points(observer, subsolar, bearing, distance_m)]
    features.extend(
        {"type": "Feature", "properties": {"kind": "bearing", "color": "#e32636", "width": 3, "popup": f"Azimuth to Sun: {bearing:.2f} degrees"}, "geometry": {"type": "LineString", "coordinates": part}}
        for part in zf.split_antimeridian_line(route)
    )
    bounds = route + [[sun_lon, sun_lat], [obs_lon, obs_lat]]
    return render_geojson_map(features, (center_lat, center_lon), bounds, zoom, pitch=40)


@callback(
    Output("sn-location-output", "children"),
    Output("sn-map-iframe", "srcDoc"),
    Output("sn-results-output", "children"),
    Input("sn-date", "value"), Input("sn-hour", "value"),
    Input("sn-minute", "value"), Input("sn-second", "value"),
    Input("sn-mgrs", "value"),
)
def update_sun_position(date_value, hour, minute, second, mgrs):
    if not mgrs:
        return "Enter an MGRS location.", "", "Enter a location and UTC date/time."
    try:
        location = zf.MGRS2LL(mgrs.strip())
        observer = float(location[1]), float(location[2])
        location_output = html.Div(
            [
                html.P(f"Latitude: {observer[0]:.5f} | Longitude: {observer[1]:.5f}", className="mb-1"),
                html.P(f"MGRS: {mgrs.strip()}", className="mb-0"),
            ]
        )
        if any(value is None for value in (date_value, hour, minute, second)):
            return location_output, "", "Enter a UTC date and time."

        selected_date = datetime.strptime(date_value, "%Y-%m-%d")
        hour, minute, second = int(hour), int(minute), int(second)
        if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59:
            raise ValueError("Enter a valid UTC time.")

        when = (selected_date.year, selected_date.month, selected_date.day, hour, minute, second, 0)
        subsolar_result = zf.subsolar(list(when[:6]))
        subsolar = float(subsolar_result[0]), float(subsolar_result[1])
        distance_m, bearing = zf.LLDist(observer[0], observer[1], subsolar[0], subsolar[1])
        elevation = float(zf.sunpos(when, observer, True)[1])
        sun_mgrs = zf.LL2MGRS(*subsolar)[1]
        observer_mgrs = zf.LL2MGRS(*observer)[1]
        map_html = _map_html(subsolar, observer, float(bearing), float(distance_m))
        results = dbc.Row(
            [
                dbc.Col([html.Strong("UTC time: "), f"{date_value} {hour:02d}:{minute:02d}:{second:02d}"], xs=12),
                dbc.Col([html.Strong("Subsolar Point: "), f"{subsolar[0]:.5f}, {subsolar[1]:.5f}"], xs=12, md=6),
                dbc.Col([html.Strong("Subsolar MGRS: "), sun_mgrs], xs=12, md=6),
                dbc.Col([html.Strong("Observer MGRS: "), observer_mgrs], xs=12, md=6),
                dbc.Col([html.Strong("Distance to Subsolar Point: "), f"{distance_m:,.0f} meters"], xs=12, md=6),
                dbc.Col([html.Strong("Azimuth to the Sun: "), f"{bearing:.2f}°"], xs=12, md=6),
                dbc.Col([html.Strong("Elevation to the Sun: "), f"{elevation:.2f}°"], xs=12, md=6),
            ], className="g-2",
        )
        return location_output, map_html, results
    except Exception as exc:
        return "Could not decode the MGRS location.", "", dbc.Alert(f"Could not calculate sun position: {exc}", color="danger", className="mb-0")
