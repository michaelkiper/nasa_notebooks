from typing import Any, List, Dict, Tuple, Union

def listify(data):
    if data is None: 
        return []
    return data if isinstance(data, (list, tuple, set)) else [data]

def tupleify(data):
    if data is None:
        return tuple()
    return tuple(data) if isinstance(data, (list, tuple, set)) else (data,)
