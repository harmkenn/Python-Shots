"""Estimate an observer's location from a celestial object's direction."""

import math
from datetime import datetime, timezone

import dash
import ephem
from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
from SubApps.maplibre_utils import render_geojson_map
from SubApps import z_functions as zf


dash.register_page(__name__, path="/h_celgrid", name="Celestial Grid", order=8)

OBJECTS = ["the Sun", "the Moon", "Polaris", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
NOW = datetime.now(timezone.utc)


layout = dbc.Container(
    [
        html.H2("Celestial Grid", className="mb-2"),
        html.P("Estimate your location from the direction and altitude of a celestial object."),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Celestial object and UTC time"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Celestial Object"),
                                        dbc.Select(id="hc-object", options=[{"label": name, "value": name} for name in OBJECTS], value="the Sun", className="mb-2"),
                                        dbc.Label("Date (UTC)"),
                                        dbc.Input(id="hc-date", type="date", value=NOW.strftime("%Y-%m-%d"), className="mb-2"),
                                        dbc.Row(
                                            [
                                                dbc.Col([dbc.Label("Hour"), dbc.Input(id="hc-hour", type="number", min=0, max=23, value=NOW.hour)], width=4),
                                                dbc.Col([dbc.Label("Minute"), dbc.Input(id="hc-minute", type="number", min=0, max=59, value=NOW.minute)], width=4),
                                                dbc.Col([dbc.Label("Second"), dbc.Input(id="hc-second", type="number", min=0, max=59, value=NOW.second)], width=4),
                                            ], className="mb-2",
                                        ),
                                        dbc.Button("Use current UTC time", id="hc-now-button", n_clicks=0, color="secondary", outline=True),
                                    ]
                                ),
                            ], className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Observed direction"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Azimuth to object (degrees)"),
                                        dbc.Input(id="hc-azimuth", type="number", min=0, max=360, value=110, step="any", className="mb-2"),
                                        dbc.Label("Altitude to object (degrees)"),
                                        dbc.Input(id="hc-altitude", type="number", min=-90, max=90, value=45, step="any"),
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
                                                dbc.Input(id="hc-lookup", type="text", placeholder="Place or address (3+ characters)"),
                                                dbc.Button("Lookup", id="hc-lookup-button", n_clicks=0),
                                            ], className="mb-2",
                                        ),
                                        html.Div(id="hc-lookup-output"),
                                    ]
                                ),
                            ], className="mb-3",
                        ),
                        dbc.Card([dbc.CardHeader("Calculated positions"), dbc.CardBody(html.Div(id="hc-results-output"))]),
                    ], xs=12, lg=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Sub-celestial point and estimated observer"),
                            dbc.CardBody(
                                html.Iframe(
                                    id="hc-map-iframe",
                                    title="Map showing the sub-celestial point and estimated observer location",
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
    Output("hc-date", "value"), Output("hc-hour", "value"),
    Output("hc-minute", "value"), Output("hc-second", "value"),
    Input("hc-now-button", "n_clicks"), prevent_initial_call=True,
)
def set_current_utc_time(n_clicks):
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d"), now.hour, now.minute, now.second


@callback(
    Output("hc-lookup-output", "children"),
    Input("hc-lookup-button", "n_clicks"), State("hc-lookup", "value"),
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


def _celestial_body(name):
    bodies = {
        "the Sun": ephem.Sun,
        "the Moon": ephem.Moon,
        "Polaris": lambda: ephem.star("Polaris"),
        "Mercury": ephem.Mercury,
        "Venus": ephem.Venus,
        "Mars": ephem.Mars,
        "Jupiter": ephem.Jupiter,
        "Saturn": ephem.Saturn,
        "Uranus": ephem.Uranus,
        "Neptune": ephem.Neptune,
    }
    return bodies[name]()


def _route_points(observer, sub_celestial, bearing, distance_m, segments=80):
    if distance_m <= 1e-6:
        return [list(observer), list(sub_celestial)]
    points = [
        list(zf.vPolar(observer[0], observer[1], bearing, distance_m * i / segments))
        for i in range(segments + 1)
    ]
    points[-1] = list(sub_celestial)
    return points


def _map_html(sub_celestial, observer, bearing, distance_m, object_name):
    cel_lat, cel_lon = sub_celestial
    obs_lat, obs_lon = observer
    lon_delta = ((obs_lon - cel_lon + 180) % 360) - 180
    center_lon = ((cel_lon + lon_delta / 2 + 180) % 360) - 180
    center_lat = (cel_lat + obs_lat) / 2
    zoom = max(2, min(15, int(-1.36 * math.log(max(distance_m / 1000, 1)) + 14)))
    cel_mgrs = zf.LL2MGRS(cel_lat, cel_lon)[1]
    observer_mgrs = zf.LL2MGRS(obs_lat, obs_lon)[1]
    features = [
        {"type": "Feature", "properties": {"kind": "sub-point", "label": f"{object_name} sub-point", "popup": f"{object_name} sub-point MGRS: {cel_mgrs}", "color": "#f59e0b", "radius": 9}, "geometry": {"type": "Point", "coordinates": [cel_lon, cel_lat]}},
        {"type": "Feature", "properties": {"kind": "observer", "label": "Estimated observer", "popup": f"Estimated observer MGRS: {observer_mgrs}", "color": "#1976d2", "radius": 8}, "geometry": {"type": "Point", "coordinates": [obs_lon, obs_lat]}},
    ]
    route = [[float(lon), float(lat)] for lat, lon in _route_points(observer, sub_celestial, bearing, distance_m)]
    features.extend(
        {"type": "Feature", "properties": {"kind": "bearing", "color": "#e32636", "width": 3, "popup": f"Azimuth to {object_name}: {bearing:.2f} degrees"}, "geometry": {"type": "LineString", "coordinates": part}}
        for part in zf.split_antimeridian_line(route)
    )
    bounds = route + [[cel_lon, cel_lat], [obs_lon, obs_lat]]
    return render_geojson_map(features, (center_lat, center_lon), bounds, zoom, pitch=40)


@callback(
    Output("hc-map-iframe", "srcDoc"), Output("hc-results-output", "children"),
    Input("hc-object", "value"), Input("hc-date", "value"),
    Input("hc-hour", "value"), Input("hc-minute", "value"), Input("hc-second", "value"),
    Input("hc-azimuth", "value"), Input("hc-altitude", "value"),
)
def update_celestial_grid(object_name, date_value, hour, minute, second, azimuth, altitude):
    values = (date_value, hour, minute, second, azimuth, altitude)
    if not object_name or any(value is None for value in values):
        return "", "Enter an object, UTC date/time, azimuth, and altitude."
    try:
        selected_date = datetime.strptime(date_value, "%Y-%m-%d")
        hour, minute, second = int(hour), int(minute), int(second)
        azimuth, altitude = float(azimuth), float(altitude)
        if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59:
            raise ValueError("Enter a valid UTC time.")
        if not 0 <= azimuth <= 360 or not -90 <= altitude <= 90:
            raise ValueError("Azimuth must be 0–360° and altitude must be −90–90°.")

        utc_time = datetime(selected_date.year, selected_date.month, selected_date.day, hour, minute, second, tzinfo=timezone.utc)
        origin = ephem.Observer()
        origin.lat, origin.lon = "0", "0"
        origin.date = ephem.Date(utc_time)
        body = _celestial_body(object_name)
        body.compute(origin)
        body_azimuth = math.degrees(float(body.az)) % 360
        body_altitude = math.degrees(float(body.alt))

        sub_celestial_distance = 40050 * (90 - body_altitude) / 360 * 1000
        sub_celestial = tuple(float(v) for v in zf.vPolar(0, 0, body_azimuth, sub_celestial_distance))
        observer_distance = 40050 * (90 - altitude) / 360 * 1000
        observer_result = zf.revVpolar(sub_celestial[0], sub_celestial[1], azimuth, observer_distance)
        observer = float(observer_result[0]), (float(observer_result[1]) + 540) % 360 - 180
        distance_check, bearing_to_object = zf.LLDist(observer[0], observer[1], sub_celestial[0], sub_celestial[1])

        map_html = _map_html(sub_celestial, observer, float(bearing_to_object), float(distance_check), object_name)
        results = dbc.Row(
            [
                dbc.Col([html.Strong("UTC time: "), utc_time.strftime("%Y-%m-%d %H:%M:%S")], xs=12),
                dbc.Col([html.Strong(f"{object_name} azimuth at 0°, 0°: "), f"{body_azimuth:.2f}°"], xs=12, md=6),
                dbc.Col([html.Strong(f"{object_name} altitude at 0°, 0°: "), f"{body_altitude:.2f}°"], xs=12, md=6),
                dbc.Col([html.Strong("Sub-celestial coordinates: "), f"{sub_celestial[0]:.5f}, {sub_celestial[1]:.5f}"], xs=12, md=6),
                dbc.Col([html.Strong("Sub-celestial MGRS: "), zf.LL2MGRS(*sub_celestial)[1]], xs=12, md=6),
                dbc.Col([html.Strong("Estimated observer coordinates: "), f"{observer[0]:.5f}, {observer[1]:.5f}"], xs=12, md=6),
                dbc.Col([html.Strong("Estimated observer MGRS: "), zf.LL2MGRS(*observer)[1]], xs=12, md=6),
                dbc.Col([html.Strong(f"Distance to {object_name}'s sub-point: "), f"{distance_check:,.0f} meters"], xs=12, md=6),
                dbc.Col([html.Strong(f"Calculated azimuth to {object_name}: "), f"{bearing_to_object:.2f}°"], xs=12, md=6),
            ], className="g-2",
        )
        return map_html, results
    except Exception as exc:
        return "", dbc.Alert(f"Could not calculate the observer position: {exc}", color="danger", className="mb-0")
