"""Find a celestial object's direction from an observer location and UTC time."""

import math
from datetime import datetime, timezone

import dash
import ephem
from dash import Input, Output, State, callback, html
import dash_bootstrap_components as dbc
from SubApps.maplibre_utils import render_geojson_map
from SubApps import z_functions as zf


dash.register_page(__name__, path="/i_celnorth", name="Celestial North", order=9)

OBJECTS = ["the Sun", "the Moon", "Polaris", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
NOW = datetime.now(timezone.utc)


layout = dbc.Container(
    [
        html.H2("Celestial North", className="mb-2"),
        html.P("Find the azimuth and altitude of a celestial object from your location."),
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
                                        dbc.Select(id="cn-object", options=[{"label": name, "value": name} for name in OBJECTS], value="the Sun", className="mb-2"),
                                        dbc.Label("Date (UTC)"),
                                        dbc.Input(id="cn-date", type="date", value=NOW.strftime("%Y-%m-%d"), className="mb-2"),
                                        dbc.Row(
                                            [
                                                dbc.Col([dbc.Label("Hour"), dbc.Input(id="cn-hour", type="number", min=0, max=23, value=NOW.hour)], width=4),
                                                dbc.Col([dbc.Label("Minute"), dbc.Input(id="cn-minute", type="number", min=0, max=59, value=NOW.minute)], width=4),
                                                dbc.Col([dbc.Label("Second"), dbc.Input(id="cn-second", type="number", min=0, max=59, value=NOW.second)], width=4),
                                            ], className="mb-2",
                                        ),
                                        dbc.Button("Use current UTC time", id="cn-now-button", n_clicks=0, color="secondary", outline=True),
                                    ]
                                ),
                            ], className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Observer location"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Location (MGRS)"),
                                        dbc.Input(id="cn-mgrs", type="text", value="32TPR9792346676", className="mb-2"),
                                        html.Div(id="cn-location-output"),
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
                                                dbc.Input(id="cn-lookup", type="text", placeholder="Place or address (3+ characters)"),
                                                dbc.Button("Lookup", id="cn-lookup-button", n_clicks=0),
                                            ], className="mb-2",
                                        ),
                                        html.Div(id="cn-lookup-output"),
                                    ]
                                ),
                            ], className="mb-3",
                        ),
                        dbc.Card([dbc.CardHeader("Celestial object position"), dbc.CardBody(html.Div(id="cn-results-output"))]),
                    ], xs=12, lg=4,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Sub-celestial point and observer"),
                            dbc.CardBody(
                                html.Iframe(
                                    id="cn-map-iframe",
                                    title="Map showing the sub-celestial point, observer, and direction to the object",
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
    Output("cn-date", "value"), Output("cn-hour", "value"),
    Output("cn-minute", "value"), Output("cn-second", "value"),
    Input("cn-now-button", "n_clicks"), prevent_initial_call=True,
)
def set_current_utc_time(n_clicks):
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d"), now.hour, now.minute, now.second


@callback(
    Output("cn-lookup-output", "children"), Output("cn-mgrs", "value"),
    Input("cn-lookup-button", "n_clicks"), State("cn-lookup", "value"),
    prevent_initial_call=True,
)
def process_lookup(n_clicks, query):
    if not n_clicks or not query or len(query.strip()) < 3:
        return "Enter at least three characters to search.", dash.no_update
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
        ), place[3]
    except Exception as exc:
        return dbc.Alert(f"Lookup error: {exc}", color="danger", className="mb-0"), dash.no_update


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


def _route_points(observer, sub_point, bearing, distance_m, segments=80):
    if distance_m <= 1e-6:
        return [list(observer), list(sub_point)]
    points = [
        list(zf.vPolar(observer[0], observer[1], bearing, distance_m * i / segments))
        for i in range(segments + 1)
    ]
    points[-1] = list(sub_point)
    return points


def _map_html(sub_point, observer, bearing, distance_m, object_name):
    point_lat, point_lon = sub_point
    obs_lat, obs_lon = observer
    lon_delta = ((obs_lon - point_lon + 180) % 360) - 180
    center_lon = ((point_lon + lon_delta / 2 + 180) % 360) - 180
    center_lat = (point_lat + obs_lat) / 2
    zoom = max(2, min(15, int(-1.36 * math.log(max(distance_m / 1000, 1)) + 14)))
    point_mgrs = zf.LL2MGRS(point_lat, point_lon)[1]
    observer_mgrs = zf.LL2MGRS(obs_lat, obs_lon)[1]
    features = [
        {"type": "Feature", "properties": {"kind": "sub-point", "label": f"{object_name} sub-point", "popup": f"{object_name} sub-point MGRS: {point_mgrs}", "color": "#f59e0b", "radius": 9}, "geometry": {"type": "Point", "coordinates": [point_lon, point_lat]}},
        {"type": "Feature", "properties": {"kind": "observer", "label": "Observer", "popup": f"Observer MGRS: {observer_mgrs}", "color": "#1976d2", "radius": 8}, "geometry": {"type": "Point", "coordinates": [obs_lon, obs_lat]}},
    ]
    route = [[float(lon), float(lat)] for lat, lon in _route_points(observer, sub_point, bearing, distance_m)]
    features.extend(
        {"type": "Feature", "properties": {"kind": "bearing", "color": "#e32636", "width": 3, "popup": f"Azimuth to {object_name}: {bearing:.2f} degrees"}, "geometry": {"type": "LineString", "coordinates": part}}
        for part in zf.split_antimeridian_line(route)
    )
    bounds = route + [[point_lon, point_lat], [obs_lon, obs_lat]]
    return render_geojson_map(features, (center_lat, center_lon), bounds, zoom, pitch=40)


@callback(
    Output("cn-location-output", "children"),
    Output("cn-map-iframe", "srcDoc"),
    Output("cn-results-output", "children"),
    Input("cn-object", "value"), Input("cn-date", "value"),
    Input("cn-hour", "value"), Input("cn-minute", "value"), Input("cn-second", "value"),
    Input("cn-mgrs", "value"),
)
def update_celestial_position(object_name, date_value, hour, minute, second, mgrs):
    if not mgrs:
        return "Enter an MGRS location.", "", "Enter a location and UTC date/time."
    try:
        location = zf.MGRS2LL(mgrs.strip())
        observer_lat, observer_lon = float(location[1]), float(location[2])
        location_output = html.Div(
            [
                html.P(f"Latitude: {observer_lat:.5f} | Longitude: {observer_lon:.5f}", className="mb-1"),
                html.P(f"MGRS: {mgrs.strip()}", className="mb-0"),
            ]
        )
        if not object_name or any(value is None for value in (date_value, hour, minute, second)):
            return location_output, "", "Enter an object and UTC date/time."

        selected_date = datetime.strptime(date_value, "%Y-%m-%d")
        hour, minute, second = int(hour), int(minute), int(second)
        if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59:
            raise ValueError("Enter a valid UTC time.")
        utc_time = datetime(selected_date.year, selected_date.month, selected_date.day, hour, minute, second, tzinfo=timezone.utc)
        origin = ephem.Observer()
        origin.lat, origin.lon = math.radians(observer_lat), math.radians(observer_lon)
        origin.date = ephem.Date(utc_time)
        body = _celestial_body(object_name)
        body.compute(origin)
        azimuth = math.degrees(float(body.az)) % 360
        altitude = math.degrees(float(body.alt))

        sub_point_distance = 40050 * (90 - altitude) / 360 * 1000
        sub_point = tuple(float(value) for value in zf.vPolar(observer_lat, observer_lon, azimuth, sub_point_distance))
        observer_mgrs = zf.LL2MGRS(observer_lat, observer_lon)[1]
        point_mgrs = zf.LL2MGRS(*sub_point)[1]
        map_html = _map_html(sub_point, (observer_lat, observer_lon), azimuth, sub_point_distance, object_name)
        results = dbc.Row(
            [
                dbc.Col([html.Strong("UTC time: "), utc_time.strftime("%Y-%m-%d %H:%M:%S")], xs=12),
                dbc.Col([html.Strong(f"Azimuth to {object_name}: "), f"{azimuth:.2f}°"], xs=12, md=6),
                dbc.Col([html.Strong(f"Altitude to {object_name}: "), f"{altitude:.2f}°"], xs=12, md=6),
                dbc.Col([html.Strong("Observer MGRS: "), observer_mgrs], xs=12, md=6),
                dbc.Col([html.Strong("Sub-celestial coordinates: "), f"{sub_point[0]:.5f}, {sub_point[1]:.5f}"], xs=12, md=6),
                dbc.Col([html.Strong("Sub-celestial MGRS: "), point_mgrs], xs=12, md=6),
                dbc.Col([html.Strong("Distance to sub-celestial point: "), f"{sub_point_distance:,.0f} meters"], xs=12, md=6),
            ], className="g-2",
        )
        return location_output, map_html, results
    except Exception as exc:
        return "Could not decode the MGRS location.", "", dbc.Alert(f"Could not calculate celestial position: {exc}", color="danger", className="mb-0")
