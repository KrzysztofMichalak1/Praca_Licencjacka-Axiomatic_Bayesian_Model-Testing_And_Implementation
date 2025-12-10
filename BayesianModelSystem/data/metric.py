"""This module contains metric functions like Haversine distance."""
import math

def haversine(p1, p2, return_unit='km'):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    lon1, lat1 = map(math.radians, p1)
    lon2, lat2 = map(math.radians, p2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    if return_unit == 'km':
        return 6371.0 * c
    elif return_unit == 'rad':
        return c
    else:
        raise ValueError("return_unit must be 'km' or 'rad'")
