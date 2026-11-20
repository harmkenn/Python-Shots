"""Find an observer location from the Sun's position and observed angles."""

import math
from datetime import datetime, timezone

import dash
from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
from SubApps.maplibre_utils import render_geojson_map
from SubApps import z_functions as zf


dash.register_page(__name__, path="/f_sungrid", name="Sun Grid", order=6)

layout = dbc.Container(
    [
        html.H2("Get Your Position from the Sun", className="mb-4"),
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
                                        dbc.Input(id="sg-date", type="date", value="2022-04-09", className="mb-2"),
                                        dbc.Row(
                                            [
                                                dbc.Col([dbc.Label("Hour"), dbc.Input(id="sg-hour", type="number", min=0, max=23, value=22)], width=4),
                                                dbc.Col([dbc.Label("Minute"), dbc.Input(id="sg-minute", type="number", min=0, max=59, value=40)], width=4),
                                                dbc.Col([dbc.Label("Second"), dbc.Input(id="sg-second", type="number", min=0, max=59, value=0)], width=4),
                                            ], className="mb-2",
                                        ),
                                        dbc.Button("Use current UTC time", id="sg-now-button", n_clicks=0, color="secondary", outline=True),
                                    ]
                                ),
                            ], className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Sun observation"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Azimuth to the Sun (degrees)"),
                                        dbc.Input(id="sg-azimuth", type="number", min=0, max=360, value=110, step="any", className="mb-2"),
                                        dbc.Label("Vertical Angle to the Sun (degrees)"),
                                        dbc.Input(id="sg-vertical-angle", type="number", min=0, max=90, value=45, step="any"),
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
                                                dbc.Input(id="sg-lookup", type="text", placeholder="Place or address (3+ characters)"),
                                                dbc.Button("Lookup", id="sg-lookup-button", n_clicks=0),
                                            ], className="mb-2",
                                        ),
                                        html.Div(id="sg-lookup-output"),
                                    ]
                                ),
                            ], className="mb-3",
                        ),
                        dbc.Card([dbc.CardHeader("Calculated coordinates"), dbc.CardBody(html.Div(id="sg-results-output"))]),
                    ], xs=12, lg=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Subsolar point and observer location"),
                            dbc.CardBody(
                                html.Iframe(
                                    id="sg-map-iframe",
                                    title="Map showing the subsolar point and observer location",
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
    Output("sg-date", "value"), Output("sg-hour", "value"),
    Output("sg-minute", "value"), Output("sg-second", "value"),
    Input("sg-now-button", "n_clicks"), prevent_initial_call=True,
)
def set_current_utc_time(n_clicks):
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d"), now.hour, now.minute, now.second


@callback(
    Output("sg-lookup-output", "children"),
    Input("sg-lookup-button", "n_clicks"), State("sg-lookup", "value"),
    prevent_initial_call=True,
)
def process_lookup(n_clicks, query):
    if not n_clicks or not query or len(query.strip()) < 3:
        return "Enter at least three characters to search."
    try:
        place = zf.lookup(query.strip())
        altitude = zf.elevation(place[1], place[2])
        return html.Div([
            html.P(place[0], className="mb-1 fw-bold"),
            html.P(f"Lat: {place[1]} | Lon: {place[2]}", className="mb-1"),
            html.P(f"MGRS: {place[3]}", className="mb-1"),
            html.P(f"Alt: {round(altitude, 2)} meters", className="mb-0"),
        ])
    except Exception as exc:
        return dbc.Alert(f"Lookup error: {exc}", color="danger", className="mb-0")


def _route_points(observer, subsolar, bearing, distance_m, segments=80):
    if distance_m <= 0:
        return [list(observer), list(subsolar)]
    points = [
        list(zf.vPolar(observer[0], observer[1], bearing, distance_m * i / segments))
        for i in range(segments + 1)
    ]
    points[-1] = list(subsolar)
    return points


def _map_html(subsolar, observer, azimuth, bearing, distance_m):
    sun_lat, sun_lon = subsolar
    obs_lat, obs_lon = observer
    lon_delta = ((obs_lon - sun_lon + 180) % 360) - 180
    center_lon = ((sun_lon + lon_delta / 2 + 180) % 360) - 180
    zoom = max(2, min(15, int(-1.36 * math.log(max(distance_m / 1000, 1)) + 14)))
    sun_mgrs, observer_mgrs = zf.LL2MGRS(sun_lat, sun_lon)[1], zf.LL2MGRS(obs_lat, obs_lon)[1]
    features = [
        {"type": "Feature", "properties": {"kind": "subsolar", "label": "Subsolar Point", "popup": f"Subsolar Point MGRS: {sun_mgrs}", "color": "#f59e0b", "radius": 9}, "geometry": {"type": "Point", "coordinates": [sun_lon, sun_lat]}},
        {"type": "Feature", "properties": {"kind": "observer", "label": "Observer", "popup": f"Observer MGRS: {observer_mgrs}", "color": "#1976d2", "radius": 8}, "geometry": {"type": "Point", "coordinates": [obs_lon, obs_lat]}},
    ]
    route = [[float(lon), float(lat)] for lat, lon in _route_points(observer, subsolar, bearing, distance_m)]
    features.extend(
        {"type": "Feature", "properties": {"kind": "bearing", "color": "#e32636", "width": 3, "popup": f"Azimuth to Sun: {azimuth:.2f} degrees"}, "geometry": {"type": "LineString", "coordinates": part}}
        for part in zf.split_antimeridian_line(route)
    )
    bounds = route + [[sun_lon, sun_lat], [obs_lon, obs_lat]]
    return render_geojson_map(features, ((sun_lat + obs_lat) / 2, center_lon), bounds, zoom, pitch=40)


@callback(
    Output("sg-map-iframe", "srcDoc"), Output("sg-results-output", "children"),
    Input("sg-date", "value"), Input("sg-hour", "value"),
    Input("sg-minute", "value"), Input("sg-second", "value"),
    Input("sg-azimuth", "value"), Input("sg-vertical-angle", "value"),
)
def update_sun_grid(date_value, hour, minute, second, azimuth, vertical_angle):
    if any(value is None for value in (date_value, hour, minute, second, azimuth, vertical_angle)):
        return "", "Enter a UTC date/time, sun azimuth, and vertical angle."
    try:
        selected_date = datetime.strptime(date_value, "%Y-%m-%d")
        hour, minute, second = int(hour), int(minute), int(second)
        azimuth, vertical_angle = float(azimuth), float(vertical_angle)
        if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59:
            raise ValueError("Enter a valid UTC time.")
        if not 0 <= azimuth <= 360 or not 0 <= vertical_angle <= 90:
            raise ValueError("Azimuth must be 0–360° and vertical angle must be 0–90°.")

        subsolar = zf.subsolar([selected_date.year, selected_date.month, selected_date.day, hour, minute, second])
        subsolar = float(subsolar[0]), float(subsolar[1])
        distance_km = (90 - vertical_angle) / 360 * 2 * 6371 * math.pi
        distance_m = distance_km * 1000
        if distance_m <= 1e-6:
            observer = subsolar
        else:
            observer_result = zf.revVpolar(subsolar[0], subsolar[1], azimuth, distance_m)
            observer = float(observer_result[0]), (float(observer_result[1]) + 540) % 360 - 180
        distance_check, bearing_to_sun = zf.LLDist(observer[0], observer[1], subsolar[0], subsolar[1])
        sun_mgrs = zf.LL2MGRS(*subsolar)[1]
        observer_mgrs = zf.LL2MGRS(*observer)[1]
        map_html = _map_html(subsolar, observer, azimuth, bearing_to_sun, distance_m)
        results = dbc.Row([
            dbc.Col([html.Strong("UTC time: "), f"{date_value} {hour:02d}:{minute:02d}:{second:02d}"], xs=12),
            dbc.Col([html.Strong("Subsolar Point: "), f"{subsolar[0]:.5f}, {subsolar[1]:.5f}"], xs=12, md=6),
            dbc.Col([html.Strong("Subsolar MGRS: "), sun_mgrs], xs=12, md=6),
            dbc.Col([html.Strong("Observer Location: "), f"{observer[0]:.5f}, {observer[1]:.5f}"], xs=12, md=6),
            dbc.Col([html.Strong("Observer MGRS: "), observer_mgrs], xs=12, md=6),
            dbc.Col([html.Strong("Distance to Subsolar Point: "), f"{distance_check:,.0f} meters"], xs=12, md=6),
            dbc.Col([html.Strong("Azimuth to Sun: "), f"{bearing_to_sun:.2f}°"], xs=12, md=6),
        ], className="g-2")
        return map_html, results
    except Exception as exc:
        return "", dbc.Alert(f"Could not calculate sun position: {exc}", color="danger", className="mb-0")
