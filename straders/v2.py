import logging
import json
import time
import math
from dacite import from_dict
from v2_types import *
import v2_calls


USERNAME = "umuero"
TOKEN = ""
SAVE = "dataV2.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("v2")

############ SAVE LOAD ############
def loadJson(fileName: str) -> State:
    try:
        with open(fileName, "r") as f:
            loadJson = json.load(f)
            return from_dict(State, loadJson)
    except Exception:
        print("no save data, continuing")
    return State()


def saveJson(fileName: str, st: State):
    with open(fileName, "w") as f:
        f.write(st.toJSON())


############ RESET LOAN ############
"""
documentations
ShipNavigation.arrivedAt 

deliver_on_contract
js.data.data
"""


def reset():
    token = ""
    while token == "":
        st = v2_calls.systems(USERNAME)
        if st.token:
            v2_calls.TOKEN = st.token
            token = st.token
        else:
            time.sleep(60)
            continue
        st.systems = v2_calls.list_systems()


"""
        availableLoans()
        availableGoods()
        availableShips()
        availableStructures()
        for glx in st.galaxies:
            updateGalaxy(glx)


def updateGalaxy(galaxy="OE"):
    js = callApi("GET", "/systems/%s/ship-listings" % galaxy)
    st.galaxies[galaxy] = {}
    if "shipListings" in js:
        for ship in js["shipListings"]:
            if "purchaseLocations" not in st.ships[ship["type"]]:
                st.ships[ship["type"]]["purchaseLocations"] = {}
            st.ships[ship["type"]]["purchaseLocations"].update({i["location"]: i["price"] for i in ship["purchaseLocations"]})
    js = callApi("GET", "/systems/%s/locations" % galaxy)
    if "locations" in js:
        st.locations.update({i["symbol"]: i for i in js["locations"]})
        for warp in js["locations"]:
            parts = warp["symbol"].split("-W-")
            if len(parts) > 1:
                currentG = parts[0]
                newW = parts[1]
                st.galaxies[currentG][newW] = [warp["symbol"]]
                for g in st.galaxies.get(newW, {}).keys():
                    if g != currentG:
                        st.galaxies[currentG][g] = st.galaxies[currentG][newW].copy()
                        st.galaxies[currentG][g].extend(st.galaxies[newW][g])
                for g in [i for i in st.galaxies.keys() if i != currentG and i != newW]:
                    if newW not in st.galaxies[g]:
                        st.galaxies[g][newW] = st.galaxies[g][currentG].copy()
                        st.galaxies[g][newW].append(warp["symbol"])
"""
