"""
z_functions.py - Tactical & Geodetic Helper Module for Dash Apps
Contains core mathematical transformations, MGRS/UTM conversions,
geodesic calculations, and placeholder integrations for elevation & lookups.
"""

import math
from typing import Tuple, List, Union, Dict, Any


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
    clean_mgrs = mgrs_str.replace(" ", "").upper()
    if len(clean_mgrs) < 5:
        raise ValueError("Invalid MGRS string length.")

    # 1. Parse GZD Zone & Band
    gzd_len = 3 if clean_mgrs[:2].isdigit() and clean_mgrs[2].isalpha() else 2
    zone_num = int(clean_mgrs[:gzd_len - 1])
    lat_band = clean_mgrs[gzd_len - 1]

    # 2. Parse 100km Square Identification
    sq_col = clean_mgrs[gzd_len]
    sq_row = clean_mgrs[gzd_len + 1]

    # 3. Parse Numerical Offsets
    num_part = clean_mgrs[gzd_len + 2:]
    half_len = len(num_part) // 2
    
    e_str = num_part[:half_len].ljust(5, '0')
    n_str = num_part[half_len:].ljust(5, '0')

    e_offset = int(e_str)
    n_offset = int(n_str)

    # 4. Reconstruct Approximate Full Easting
    col_set = (zone_num - 1) % 3
    col_letters = ["ABCDEFGH", "JKLMNPQR", "STUVWXYZ"][col_set]
    col_idx = col_letters.index(sq_col) + 1
    easting = (col_idx * 100000) + e_offset

    # 5. Reconstruct Approximate Full Northing
    row_letters_odd = "ABCDEFGHJKLMNPQRSTUV"
    row_letters_even = "FGHJKLMNPQRSTUVABCDE"
    row_letters = row_letters_even if (zone_num % 2 == 0) else row_letters_odd
    row_idx = row_letters.index(sq_row)

    # Approximate Latitude based on Band Letter
    band_letters = "CDEFGHJKLMNPQRSTUVWX"
    approx_lat = (band_letters.index(lat_band) * 8) - 80 + 4
    
    # Calculate approximate central northing block
    approx_northing = 0.0 if approx_lat >= 0 else 10000000.0
    if approx_lat >= 0:
        approx_northing += approx_lat * 111000.0

    northing_block = math.floor(approx_northing / 2000000.0) * 2000000.0
    northing = northing_block + (row_idx * 100000) + n_offset

    # Reverse UTM to Lat/Lon Inverse Transformation
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)
    e_prime2 = e2 / (1 - e2)
    k0 = 0.9996

    x = easting - 500000.0
    y = northing
    if approx_lat < 0 and y < 10000000.0:
        y -= 10000000.0  # Adjust Southern Hemisphere offset if needed

    M = y / k0
    mu = M / (a * (1 - e2/4 - 3*e2**2/64 - 5*e2**3/256))

    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    phi1 = mu + (3*e1/2 - 27*e1**3/32)*math.sin(2*mu) + (21*e1**2/16 - 55*e1**4/32)*math.sin(4*mu)

    sin_phi1 = math.sin(phi1)
    cos_phi1 = math.cos(phi1)
    tan_phi1 = math.tan(phi1)

    N1 = a / math.sqrt(1 - e2 * sin_phi1**2)
    R1 = a * (1 - e2) / ((1 - e2 * sin_phi1**2)**1.5)
    T1 = tan_phi1**2
    C1 = e_prime2 * cos_phi1**2
    D = x / (N1 * k0)

    lat = phi1 - (N1 * tan_phi1 / R1) * (
        D**2/2 - (5 + 3*T1 + 10*C1 - 4*C1**2 - 9*e_prime2)*(D**4)/24
        + (61 + 90*T1 + 298*C1 + 45*T1**2 - 252*e_prime2)*(D**6)/720
    )
    
    central_meridian = math.radians((zone_num - 1) * 6 - 180 + 3)
    lon = central_meridian + (
        D - (1 + 2*T1 + C1)*(D**3)/6
        + (5 - 2*C1 + 28*T1 - 3*C1**2 + 8*e_prime2 + 24*T1**2)*(D**5)/120
    ) / cos_phi1

    utm_formatted = f"{zone_num:02d}{lat_band} {int(easting):06d}mE {int(northing):07d}mN"
    return utm_formatted, math.degrees(lat), math.degrees(lon)


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
# 3. Placeholders for Database / Remote API Lookups
# =====================================================================

def lookup(query_str: str) -> List[Any]:
    """
    Search lookup function for named points/targets.
    Replace with actual database lookup as needed.
    
    :return: [Name, Latitude, Longitude, MGRS_String]
    """
    query_clean = query_str.strip().upper()

    # Pre-populated mock data store
    mock_db = {
        "ALPHA": ["Point Alpha", -30.0, 30.0, "36JUU0000000000"],
        "BRAVO": ["Point Bravo", -29.9, 30.1, "36JUU1012301234"],
        "CHARLIE": ["Point Charlie", -30.1, 29.9, "36JTT8987689876"],
    }

    if query_clean in mock_db:
        return mock_db[query_clean]

    # Dynamic fallback based on input
    utm, mgrs = LL2MGRS(-30.0, 30.0)
    return [f"Target {query_str}", -30.0, 30.0, mgrs]


def elevation(lat: float, lon: float) -> float:
    """
    Fetches terrain elevation in meters for given coordinates.
    Replace with local DEM raster query or USGS/Open-Elevation API integration.
    """
    # Placeholder return value in meters
    return 150.0
