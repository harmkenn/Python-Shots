"""
z_functions.py - Tactical & Geodetic Helper Module for Dash Apps
Contains core mathematical transformations, MGRS/UTM conversions,
geodesic calculations, Google place lookup, and placeholder elevation data.
"""

import json
import math
import os
import re
from pathlib import Path
from typing import Tuple, List, Union, Dict, Any
from urllib.parse import urlencode
from urllib.request import urlopen

import ephem
import pandas as pd
from numpy import arcsin, arctan, arctan2, cos, exp, pi, radians, sin, sqrt, tan
from sklearn.ensemble import GradientBoostingRegressor

from pygeodesy.ellipsoidalKarney import LatLon
from pygeodesy.mgrs import parseMGRS


# =====================================================================
# 1. Coordinate Conversion: Lat/Lon <-> UTM & MGRS (WGS84 Ellipsoid)
# =====================================================================

def LL2MGRS(lat: float, lon: float, precision: int = 5) -> Tuple[str, str]:
    """
    Converts WGS84 Latitude & Longitude to UTM and MGRS coordinate strings.
    
    :param lat: Latitude in decimal degrees.
    :param lon: Longitude in decimal degrees.
    :param precision: Digits per axis in MGRS (5 digits = 1m precision).
    :return: Tuple containing (UTM string, MGRS string)
    """
    # WGS84 Ellipsoid Constants
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = f * (2 - f)
    e_prime2 = e2 / (1 - e2)
    k0 = 0.9996

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # 1. Determine UTM Zone Number & Central Meridian
    zone_num = int((lon + 180) // 6) + 1

    # Special UTM Zone Overrides (Norway / Svalbard)
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone_num = 32
    elif 72.0 <= lat < 84.0:
        if 0.0 <= lon < 9.0:
            zone_num = 31
        elif 9.0 <= lon < 21.0:
            zone_num = 33
        elif 21.0 <= lon < 33.0:
            zone_num = 35
        elif 33.0 <= lon < 42.0:
            zone_num = 37

    central_meridian = math.radians((zone_num - 1) * 6 - 180 + 3)

    # 2. Redfearn Projection Formulas
    dlon = lon_rad - central_meridian
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    tan_lat = math.tan(lat_rad)

    N = a / math.sqrt(1 - e2 * sin_lat**2)
    T = tan_lat**2
    C = e_prime2 * cos_lat**2
    A = dlon * cos_lat

    # Meridional Arc (M)
    M = a * (
        (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256) * lat_rad
        - (3*e2/8 + 3*e2**2/32 + 45*e2**3/1024) * math.sin(2 * lat_rad)
        + (15*e2**2/256 + 45*e2**3/1024) * math.sin(4 * lat_rad)
        - (35*e2**3/3072) * math.sin(6 * lat_rad)
    )

    # Compute Easting & Northing
    easting = 500000.0 + k0 * N * (
        A
        + (1 - T + C) * (A**3) / 6.0
        + (5 - 18*T + T**2 + 72*C - 58*e_prime2) * (A**5) / 120.0
    )

    northing = k0 * (
        M
        + N * tan_lat * (
            (A**2) / 2.0
            + (5 - T + 9*C + 4*C**2) * (A**4) / 24.0
            + (61 - 58*T + T**2 + 600*C - 330*e_prime2) * (A**6) / 720.0
        )
    )

    if lat < 0:
        northing += 10000000.0  # False Northing for Southern Hemisphere

    e_int = int(math.floor(easting))
    n_int = int(math.floor(northing))

    # 3. Determine Latitude Band Letter
    band_letters = "CDEFGHJKLMNPQRSTUVWX"
    if -80 <= lat <= 84:
        band_idx = int((lat + 80) // 8)
        if band_idx >= len(band_letters):
            band_idx = len(band_letters) - 1
        lat_band = band_letters[band_idx]
    else:
        lat_band = 'Z'

    # 4. Determine 100km Square Identifiers
    col_set = (zone_num - 1) % 3
    col_letters = [
        "ABCDEFGH",  # Set 1
        "JKLMNPQR",  # Set 2
        "STUVWXYZ"   # Set 3
    ][col_set]
    col_idx = int(e_int // 100000) - 1
    sq_col = col_letters[col_idx % len(col_letters)]

    row_letters_odd = "ABCDEFGHJKLMNPQRSTUV"
    row_letters_even = "FGHJKLMNPQRSTUVABCDE"
    row_letters = row_letters_even if (zone_num % 2 == 0) else row_letters_odd
    row_idx = int((n_int % 2000000) // 100000)
    sq_row = row_letters[row_idx % len(row_letters)]

    # 5. Build Result Strings
    gzd = f"{zone_num:02d}{lat_band}"
    sq_id = f"{sq_col}{sq_row}"

    e_offset = int((e_int % 100000) // (10 ** (5 - precision)))
    n_offset = int((n_int % 100000) // (10 ** (5 - precision)))
    grid_coords = f"{e_offset:0{precision}d}{n_offset:0{precision}d}"

    utm_str = f"{gzd} {e_int:06d}mE {n_int:07d}mN"
    mgrs_str = f"{gzd}{sq_id}{grid_coords}"

    return utm_str, mgrs_str


def MGRS2LL(mgrs_str: str) -> Tuple[str, float, float]:
    """
    Converts an MGRS string into (UTM formatted string, Latitude, Longitude).
    Supports 10-digit, 8-digit, 6-digit, and 4-digit MGRS strings.
    """
    clean_mgrs = "".join(mgrs_str.split()).upper()
    if not re.fullmatch(
        r"(?:0?[1-9]|[1-5]\d|60)[C-HJ-NP-X][A-HJ-NP-Z][A-HJ-NP-V](?:\d{2}){0,5}",
        clean_mgrs,
    ):
        raise ValueError("Invalid MGRS string.")
    if len(clean_mgrs) == 5:
        clean_mgrs += "0000000000"

    mgrs = parseMGRS(clean_mgrs)
    point = mgrs.toLatLon(LatLon=LatLon, center=False)
    utm = mgrs.toUtm(center=False)
    utm_formatted = (
        f"{mgrs.zone:02d}{mgrs.band} "
        f"{int(utm.easting):06d}mE {int(utm.northing):07d}mN"
    )
    return utm_formatted, point.lat, point.lon


# =====================================================================
# 2. Geodesic Calculations: Distance, Bearing & Polar Projections
# =====================================================================

def LLDist(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    """
    Calculates Geodesic Great-Circle Distance (in meters) and Initial Bearing (in degrees)
    between two point coordinates using the Haversine formula.
    
    :return: Tuple containing (distance_meters, initial_bearing_degrees)
    """
    R = 6371000.0  # Earth radius in meters

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Distance calculation
    a = (math.sin(delta_phi / 2) ** 2) + (
        math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2) ** 2)
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    # Initial bearing calculation
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - (
        math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    )
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

    return distance, bearing


def vPolar(lat: float, lon: float, bearing_deg: float, dist_m: float) -> Tuple[float, float]:
    """
    Calculates destination point given a starting Lat/Lon, initial bearing (deg), and distance (m).
    
    :return: Tuple containing (destination_lat, destination_lon)
    """
    R = 6371000.0  # Earth radius in meters
    delta = dist_m / R
    theta = math.radians(bearing_deg)

    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)

    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta)
        + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    )

    return math.degrees(phi2), (math.degrees(lambda2) + 540) % 360 - 180


# =====================================================================
# 3. Google Geocoding Lookup
# =====================================================================

def lookup(query_str: str) -> List[Any]:
    """
    Find a place using the Google Geocoding API.

    :return: [Name, Latitude, Longitude, MGRS_String]
    """
    query = query_str.strip()
    if not query:
        raise ValueError("Enter a place or address to search.")

    api_key = os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set the GOOGLE_MAPS_API_KEY or GOOGLE_API_KEY environment variable."
        )

    params = urlencode({"address": query, "key": api_key})
    url = f"https://maps.googleapis.com/maps/api/geocode/json?{params}"
    with urlopen(url, timeout=10) as response:
        payload = json.load(response)

    status = payload.get("status")
    if status != "OK":
        detail = payload.get("error_message")
        message = f"Google Geocoding API returned {status or 'an invalid response'}"
        if detail:
            message += f": {detail}"
        raise ValueError(message)

    results = payload.get("results") or []
    if not results:
        raise ValueError("Google returned no matching locations.")

    result = results[0]
    location = result["geometry"]["location"]
    lat = float(location["lat"])
    lon = float(location["lng"])
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Google returned coordinates outside valid latitude/longitude ranges.")

    address = result.get("formatted_address", query)
    mgrs = LL2MGRS(lat, lon)[1]
    return [address, lat, lon, mgrs]


def elevation(lat: float, lon: float) -> float:
    """
    Fetches terrain elevation in meters for given coordinates.
    Replace with local DEM raster query or USGS/Open-Elevation API integration.
    """
    # Placeholder return value in meters
    return 150.0

def sunpos(when, location, refraction):
# Extract the passed data
    year, month, day, hour, minute, second, timezone = when
    latitude, longitude = location
# Math typing shortcuts

# Convert latitude and longitude to radians
    rlat = latitude*pi/180
    rlon = longitude*pi/180
# Decimal hour of the day at Greenwich
    greenwichtime = hour - timezone + minute / 60 + second / 3600
# Days from J2000, accurate from 1901 to 2099
    daynum = (
        367 * year
        - 7 * (year + (month + 9) // 12) // 4
        + 275 * month // 9
        + day
        - 730531.5
        + greenwichtime / 24
    )
# Mean longitude of the sun
    mean_long = daynum * 0.01720279239 + 4.894967873
# Mean anomaly of the Sun
    mean_anom = daynum * 0.01720197034 + 6.240040768
# Ecliptic longitude of the sun
    eclip_long = (
        mean_long
        + 0.03342305518 * sin(mean_anom)
        + 0.0003490658504 * sin(2 * mean_anom)
    )
# Obliquity of the ecliptic
    obliquity = 0.4090877234 - 0.000000006981317008 * daynum
# Right ascension of the sun
    rasc = arctan2(cos(obliquity) * sin(eclip_long), cos(eclip_long))
# Declination of the sun
    decl = arcsin(sin(obliquity) * sin(eclip_long))
# Local sidereal time
    sidereal = 4.894961213 + 6.300388099 * daynum + rlon
# Hour angle of the sun
    hour_ang = sidereal - rasc
# Local elevation of the sun
    elevation = arcsin(sin(decl) * sin(rlat) + cos(decl) * cos(rlat) * cos(hour_ang))
# Local azimuth of the sun
    azimuth = arctan2(
        -cos(decl) * cos(rlat) * sin(hour_ang),
        sin(decl) - sin(rlat) * sin(elevation)
    )
# Convert azimuth and elevation to degrees
    azimuth = into_range(azimuth*180/pi, 0, 360)
    elevation = into_range(elevation*180/pi, -180, 180)
# Refraction correction (optional)
    if refraction:
        targ = (elevation + (10.3 / (elevation + 5.11)))*pi/180
        elevation += (1.02 / tan(targ)) / 60
# Return azimuth and elevation in degrees
    return (round(azimuth, 2), round(elevation, 2))
def into_range(x, range_min, range_max):
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min

def subsolar(utc):
    ye, mo, da, ho, mi, se = utc 
    ta = pi * 2
    ut = ho + mi / 60 + se / 3600
    t = 367 * ye - 7 * (ye + (mo + 9) // 12) // 4
    dn = t + 275 * mo // 9 + da - 730531.5 + ut / 24
    sl = dn * 0.01720279239 + 4.894967873
    sa = dn * 0.01720197034 + 6.240040768
    t = sl + 0.03342305518 * sin(sa)
    ec = t + 0.0003490658504 * sin(2 * sa)
    ob = 0.4090877234 - 0.000000006981317008 * dn
    st = 4.894961213 + 6.300388099 * dn
    ra = arctan2(cos(ob) * sin(ec), cos(ec))
    de = arcsin(sin(ob) * sin(ec))
    la = de*180/pi
    lo = (ra - st)*180/pi % 360
    lo = lo - 360 if lo > 180 else lo
    return [round(la, 6), round(lo, 6)]

def sub_cel(cel_ob):
    observer = ephem.Observer()
    observer.lat = '0'  # Example latitude (London, United Kingdom)
    observer.lon = '0'
    if cel_ob == 'the Sun': 
        cel = ephem.Sun()
    elif cel_ob == 'the Moon': 
        cel = ephem.Moon()
    elif cel_ob == 'Polaris': 
        cel = ephem.star('Polaris') # https://theskylive.com/sky/stars/polaris-alpha-ursae-minoris-star
    elif cel_ob == 'Mercury': 
        cel = ephem.Mercury() # https://theskylive.com/where-is-mercury
    elif cel_ob == 'Venus': 
        cel = ephem.Venus()
    elif cel_ob == 'Mars': 
        cel = ephem.Mars()
    elif cel_ob == 'Jupiter': 
        cel = ephem.Jupiter()
    elif cel_ob == 'Saturn': 
        cel = ephem.Saturn()
    elif cel_ob == 'Uranus': 
        cel = ephem.Uranus()
    elif cel_ob == 'Neptune': 
        cel = ephem.Neptune()
    else: out = 'Something else'
    
    cel.compute(observer)
    cel_azimuth = float(cel.az)*180/pi
    cel_altitude = float(cel.alt)*180/pi
    
    sub_cel_dist = 40050*(90-cel_altitude)*1000/360
    
    sub_cel = vPolar(0,0,cel_azimuth,sub_cel_dist)
    return [cel_ob,sub_cel[1],sub_cel[0]]

def _legacy_ll_dist(lat1d,lon1d,lat2d,lon2d):
    # https://en.wikipedia.org/wiki/Vincenty%27s_formulae
    a=6378137.0
    f = 1/298.257223563
    b = (1-f)*a
    lat1r=lat1d*pi/180
    lon1r=lon1d*pi/180
    lat2r=lat2d*pi/180
    lon2r=lon2d*pi/180
    U1 = arctan((1-f)*tan(lat1r))
    U2 = arctan((1-f)*tan(lat2r))
    L = lon2r-lon1r
    lam = L

    for i in range(7):
        sinσ = sqrt((cos(U2)*sin(lam))**2+(cos(U1)*sin(U2)-sin(U1)*cos(U2)*cos(lam))**2)
        cosσ = sin(U1)*sin(U2)+cos(U1)*cos(U2)*cos(lam)
        σ = arctan2(sinσ,cosσ)
        sinα = cos(U1)*cos(U2)*sin(lam) / sinσ
        cos2α = 1 - sinα**2
        if cos2α == 0: cos2α = .000000000001
        cos2σm = cosσ - 2*sin(U1)*sin(U2) / cos2α
        C = f/16*cos2α*(4+f*(4-3*cos2α))
        lam = L+(1-C)*f*sinα*(σ+C*sinσ*(cos2σm+C*cosσ*(-1+2*cos2σm**2)))

    u2 = cos2α*(a**2/b**2-1)
    A = 1+u2/16384*(4096+u2*(-768+u2*(320-175*u2)))
    B = u2/1024*(256+u2*(-128+u2*(74-47*u2)))
    Δσ = B*sinσ*(cos2σm+B/4*(cosσ*(-1+2*cos2σm**2)-B/6*cos2σm*(-3+4*sinσ**2)*(-3+4*cos2σm**2)))
    dist = b*A*(σ-Δσ)

    faz = arctan2(sin(lon2r-lon1r)*cos(lat2r),cos(lat1r)*sin(lat2r)-sin(lat1r)*cos(lat2r)*cos(lon2r-lon1r))
    fazd = (faz*180/pi+360)%360
    baz = arctan2(sin(lon1r-lon2r)*cos(lat1r),cos(lat2r)*sin(lat1r)-sin(lat2r)*cos(lat1r)*cos(lon1r-lon2r))
    bazd = (baz*180/pi+360)%360
    iazd = (bazd+180)%360

    return [dist,fazd,bazd,iazd]

def _legacy_vpolar(lat1d,lon1d,dird,dists):
    # https://en.wikipedia.org/wiki/Vincenty%27s_formulae
    a=6378137.0
    f = 1/298.257223563
    b = (1-f)*a
 
    lat1r=lat1d*pi/180
    lon1r=lon1d*pi/180
    dirr = dird*pi/180
    U1 = arctan((1-f)*tan(lat1r))
    sig1 = arctan2(tan(U1),cos(dirr))
    sinα = cos(U1)*sin(dirr)
    u2 = (1-sinα**2)*(a**2/b**2-1)
    k1 = (sqrt(1+u2)-1)/(sqrt(1+u2)+1)
    A = (1+.25*k1**2)/(1-k1)
    B = k1*(1-3/8*k1**2)
    sig = dists/b/A

    for i in range(4):
        tsm = 2*sig1+sig
        dsig = B*sin(sig)*(cos(tsm)+B/4*(cos(sig)*(-1+2*cos(tsm))-B/6*cos(tsm)*(-3+4*sin(sig)**2)*(-3+4*cos(tsm)**2)))
        sig = dists/b/A+dsig

    lat2r = arctan2(sin(U1)*cos(sig)+cos(U1)*sin(sig)*cos(dirr),(1-f)*sqrt(sinα**2+(sin(U1)*sin(sig)-cos(U1)*cos(sig)*cos(dirr))**2))
    lam = arctan2(sin(sig)*sin(dirr),cos(U1)*cos(sig)-sin(U1)*sin(sig)*cos(dirr))
    C = f/16*(1-sinα**2)*(4+f*(4-3*(1-sinα**2)))
    L = lam-(1-C)*f*sinα*(sig+C*sin(sig)*(cos(tsm)+C*cos(sig)*(-1+2*cos(tsm)**2)))
    lon2r = L+lon1r
    lat2d = lat2r*180/pi
    lon2d = lon2r*180/pi
    if lon2d < -180: lon2d = lon2d+360
    more = _legacy_ll_dist(lat1d,lon1d,lat2d,lon2d)
    iazd = more[3]
    return [lat2d,lon2d,dird,iazd]

def revVpolar(lat2d,lon2d,lazd,distm):
    bazd = (lazd + 180)%360
    taz = bazd
    for i in range(20):
        temp = _legacy_vpolar(lat2d , lon2d,taz,distm)
        taz = taz - (temp[3]-bazd)
    lat1d = temp[0]
    lon1d = temp[1]
    liazr = (taz +180)%360
    return [lat1d,lon1d, lazd, liazr]

# Function to calculate gravitational acceleration by latitude
def gravity_by_latitude(latitude, gp = 9.832, alpha=0.0053024):
    latitude_rad = radians(latitude)
    return gp * (1 - alpha * cos(latitude_rad)**2)

# Function to calculate air density with altitude
def air_density_with_altitude(h, rho0 = 1.225, H=8500):
    return rho0 * exp(-h / H)
     
# Function to calculate the Magnus force
def magnus_force(v, spin_velocity, rho, area, S = 0.0009):
    # CL is the lift coefficient, which is dimensionless
    # spin_rate is in rad/s to be compatible with the velocity v in m/s
    CL = S * (spin_velocity / v)
    # Magnus force calculation, with v in m/s and area in m^2
    return CL * 0.5 * rho * area * v**2

# Function to calculate projectile motion with drag, varying air density, varying gravity
def projectile_motion_with_drag(v0, angle, mass, drag_coefficient, area, latitude, azimuth, initial_altitude, target_altitude, spin_velocity):
    # Convert angle to radians
    angle_rad = radians(angle)
    
    # Initial velocity components
    v0x = v0 * cos(angle_rad)
    v0y = v0 * sin(angle_rad)
    # Velocity components
    vx = v0x
    vy = v0y
    vz = 0
    
    # Time step for simulation
    dt = 0.01
    t = 0
    x = 0
    y = initial_altitude  # Set the initial altitude
    z = 0  # z-axis for lateral drift
    
    # Lists to store the trajectory points
    t_time = []
    x_points = []
    y_points = []
    z_points = []  # z-axis for lateral drift
    vx_comp = []
    vy_comp = []
    vz_comp = []
    v_mag = []
    qe = []

    # Store the initial points
    t_time.append(t) # seconds
    x_points.append(x) # meters
    y_points.append(y) # meters
    z_points.append(z)  # z-axis for lateral drift
    vx_comp.append(v0x) # m/s
    vy_comp.append(v0y) # m/s
    vz_comp.append(0) # m/s
    v_mag.append(v0) # m/s
    qe.append(angle/180*3200) # trajectory in mils

    
    
    # Calculate the gravitational acceleration for the given latitude
    g = gravity_by_latitude(latitude)
    
    # Simulation loop
    while y >= 0:
        # Calculate the air density at current altitude
        rho = air_density_with_altitude(y)

        # Calculate the Magnus force
        # The velocity v is the magnitude of the velocity vector, in m/s
        F_magnus = magnus_force(sqrt(vx**2 + vy**2), spin_velocity, rho, area)
        
        # Calculate the drag force
        F_drag_x = - (drag_coefficient * rho * area * vx**2) / (2 * mass)
        F_drag_y = - (drag_coefficient * rho * area * vy**2) / (2 * mass)
        F_drag_z = - (drag_coefficient * rho * area * vz**2) / (2 * mass)
        F_total_z = F_magnus + F_drag_z
                   
        # Update velocities
        vx += F_drag_x * dt
        vy += (F_drag_y - g) * dt
        vz += F_total_z * dt / mass
        
        # Update positions
        t += dt
        x += vx * dt
        y += vy * dt
        z += vz * dt
        
        # Store the points
        t_time.append(t) # seconds
        x_points.append(x)
        y_points.append(y)
        z_points.append(z)  # z-axis for lateral drift
        vx_comp.append(vx) #m/s
        vy_comp.append(vy)# m/s
        vz_comp.append(vz)
        v_mag.append(sqrt(vx**2+vy**2)) # m/s
        qe.append(arctan(vy/vx)/pi*3200) #in mils
        
        # Check if the projectile has reached the target altitude
        if y <= target_altitude and vy < 0:
            break
        
                  
    return t, x, y, z, t_time,x_points, y_points, z_points,vx_comp,vy_comp,vz_comp,v_mag,qe

# ML QE to Range computation

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

mlQE2Range = pd.read_csv(DATA_DIR / "2025ArtilleryData.csv")
# Split data into features (X) and target (y)
X = mlQE2Range.iloc[:,:8]
r = mlQE2Range['range']
M = mlQE2Range['Max Ord']
T = mlQE2Range['TOF']
D = mlQE2Range['Drift']

# Set regression model parameters
params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",  # Least squares loss (you can adjust this)
}

# Initialize the Gradient Boosting Regressor
reg_r = GradientBoostingRegressor(**params)
reg_M = GradientBoostingRegressor(**params)
reg_T = GradientBoostingRegressor(**params)
reg_D = GradientBoostingRegressor(**params)

# Fit the model to the training data
reg_r.fit(X, r)
reg_M.fit(X, M)
reg_T.fit(X, T)
reg_D.fit(X, D)

def _model_input(training_frame, values):
    """Build a one-row float frame matching the model's named features."""
    feature_names = training_frame.columns[:len(values)]
    return pd.DataFrame([[float(value) for value in values]], columns=feature_names, dtype=float)


def MLshot(l_lat,l_alt,i_alt,mass,GTAz,AOL,QE,v0):
    shot = _model_input(mlQE2Range, [l_lat,l_alt,i_alt,mass,GTAz,AOL,QE,v0])
    r_pred = reg_r.predict(shot)
    M_pred = reg_M.predict(shot)
    T_pred = reg_T.predict(shot)
    D_pred = reg_D.predict(shot)
    defl = (D_pred[0]-float(GTAz)+float(AOL)+3200)%6400
    return r_pred[0],M_pred[0],T_pred[0],D_pred[0],defl

# ML Range to QE computation

mlRange2QE = pd.read_csv(DATA_DIR / "2025AFATDSData.csv")
mlRange2QE_H = mlRange2QE[mlRange2QE['QE']>800]
mlRange2QE_L = mlRange2QE[mlRange2QE['QE']<800]
# Split data into features (X) and target (y)
X_H = mlRange2QE_H.iloc[:,:8]
q_H = mlRange2QE_H['QE']
M_H = mlRange2QE_H['Max Ord']
T_H = mlRange2QE_H['TOF']
D_H = mlRange2QE_H['Drift']
X_L = mlRange2QE_L.iloc[:,:8]
q_L = mlRange2QE_L['QE']
M_L = mlRange2QE_L['Max Ord']
T_L = mlRange2QE_L['TOF']
D_L = mlRange2QE_L['Drift']

# Set regression model parameters
params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",  # Least squares loss (you can adjust this)
}

# Initialize the Gradient Boosting Regressor
freg_q_H = GradientBoostingRegressor(**params)
freg_M_H = GradientBoostingRegressor(**params)
freg_T_H = GradientBoostingRegressor(**params)
freg_D_H = GradientBoostingRegressor(**params)
freg_q_L = GradientBoostingRegressor(**params)
freg_M_L = GradientBoostingRegressor(**params)
freg_T_L = GradientBoostingRegressor(**params)
freg_D_L = GradientBoostingRegressor(**params)


# Fit the model to the training data
freg_q_H.fit(X_H, q_H)
freg_M_H.fit(X_H, M_H)
freg_T_H.fit(X_H, T_H)
freg_D_H.fit(X_H, D_H)
freg_q_L.fit(X_L, q_L)
freg_M_L.fit(X_L, M_L)
freg_T_L.fit(X_L, T_L)
freg_D_L.fit(X_L, D_L)

def MLAHshot(l_lat,l_alt,i_alt,mass,GTAz,AOL,v0,range):
    shot = _model_input(mlRange2QE_H, [l_lat,l_alt,i_alt,mass,GTAz,AOL,v0,range])
    q_pred = freg_q_H.predict(shot)
    M_pred = freg_M_H.predict(shot)
    T_pred = freg_T_H.predict(shot)
    D_pred = freg_D_H.predict(shot)
    defl = (D_pred[0]-float(GTAz)+float(AOL)+3200)%6400
    return q_pred[0],M_pred[0],T_pred[0],D_pred[0],defl
def MLALshot(l_lat,l_alt,i_alt,mass,GTAz,AOL,v0,range):
    shot = _model_input(mlRange2QE_L, [l_lat,l_alt,i_alt,mass,GTAz,AOL,v0,range])
    q_pred = freg_q_L.predict(shot)
    M_pred = freg_M_L.predict(shot)
    T_pred = freg_T_L.predict(shot)
    D_pred = freg_D_L.predict(shot)
    defl = (D_pred[0]-float(GTAz)+float(AOL)+3200)%6400
    return q_pred[0],M_pred[0],T_pred[0],D_pred[0],defl
