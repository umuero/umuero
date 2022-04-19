import argparse
import logging
import json
import time
import math
from dataclasses import asdict
from dacite import from_dict
from v2base import *


USERNAME = "umuero"
TOKEN = ""
SAVE = "dataV2.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("v2")

############ SAVE LOAD ############
def loadJson(fileName: str = SAVE) -> State:
    try:
        with open(fileName, "r") as f:
            loadJson = json.load(f)
            return from_dict(State, loadJson)
    except Exception as e:
        logger.exception("no save data, continuing")
    return None


def saveJson(st: State, fileName: str = SAVE):
    with open(fileName, "w") as f:
        json.dump(asdict(st), f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="SpaceTradersV2")
    parser.add_argument("-i", "--info", help="print json and exit", action="store_true")
    parser.add_argument("-x", "--extra", help="print extra info and exit", action="store_true")
    parser.add_argument("-l", "--leader", help="print leaders and exit", action="store_true")
    parser.add_argument("-s", "--sleep", help="sleep seconds", default=60, type=int)
    parser.add_argument("--username", help="set username")
    args = parser.parse_args()
    if args.username:
        USERNAME = args.username

    st = loadJson()
    if st is None:
        v2 = V2("")
        st = v2.agent_register(USERNAME)
        v2.token = st.token
    v2 = V2(st.token)

    if args.info:
        logger.info(asdict(st.agent))
        logger.info("== SHIPS ==")
        for ship in st.ships:
            logger.info(asdict(ship))
        logger.info("== CONTRACTS ==")
        for contract in st.contracts:
            logger.info(asdict(contract))
        if args.extra:
            logger.info("== SYSTEMS ==")
            for system in st.systems.values():
                logger.info(asdict(system))
        return

    if args.leader:
        v2.leaderboard()
        return

    loopCtr = 1
    while loopCtr > 0:
        try:
            st.agent = v2.my_agent()
            st.contracts = v2.contract_list()
            st.ships = v2.ship_list()
            if args.extra:  # or loopCtr % 60 == 0:
                # daha seyrek cekmek lazim
                st.systems = {s.symbol: s for s in v2.system_list()}
                st.availableShips = []
                for system in st.systems.values():
                    if system.charted:
                        for marketNode in v2.market_list(system.symbol):
                            if marketNode not in st.markets:
                                st.markets[marketNode] = None
                        for shipyardNode in v2.shipyard_list(system.symbol):
                            try:
                                st.availableShips.extend(v2.shipyard_listing(system.symbol, shipyardNode.symbol))
                            except Exception:
                                logger.info("not charted waypoint ??")

            # if role == command ?? -> none butun marketleri gez, save et
            # if role == solar && extractor -> OE'de extract et, sat

            # contract delivery icin orbit kafi,
            # market icin dock olacak

            # if in marketNode --> refuel

            # X1-OE de gez, survey extract -> sat ??
            # her navigate result -> st.cooldowns.append
            # her survey extract scan result -> st.cooldowns.append

            saveJson(st)
        except Exception as e:
            logger.exception("main loop failed")

        loopCtr += 1
        print("sleeping", args.sleep)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()


"""
    def distance(self, locationA: Location, locationB: Location):
        dx = abs(locationA.x-locationB.x)
        dy = abs(locationA.y-locationB.y)
        sq = sqrt(dx*dx+dy*dy)
        return round(sq)

    def flightTime(self, a: Location, b: Location, speed=1):
        return round(self.distance(a, b)*(3/speed)+30)
        
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


"""
In [70]: st.ships[0]
Out[70]: Ship(symbol='UMUERO-1', frame='FRAME_FRIGATE', reactor='REACTOR_FUSION_I', engine='ENGINE_ION_DRIVE_II', fuel=1200, modules=['MODULE_GAS_TANK', 'MODULE_CARGO_HOLD', 'MODULE_CARGO_HOLD', 'MODULE_CREW_QUARTERS', 'MODULE_ENVOY_QUARTERS', 'MODULE_JUMP_DRIVE_I'], mounts=['MOUNT_SENSOR_ARRAY_II', 'MOUNT_MINING_LASER_II', 'MOUNT_GAS_SIPHON_II'], registration=Registration(factionSymbol='COMMERCE_REPUBLIC', agentSymbol='UMUERO', fee=0, role='COMMAND'), integrity=Integrity(frame=1, reactor=1, engine=1), stats=ShipStats(fuelTank=1200, cargoLimit=280, jumpRange=20), status='DOCKED', location='X1-OE-PM', cargo=[])

In [71]: st.contracts[0]
Out[71]: Contract(id='cl263cjvy001401s6alc25vab', faction='COMMERCE_REPUBLIC', type='PROCUREMENT', terms=Term(deadline='2022-05-03T11:57:12.235Z', payment=Payment(onAccepted=30000, onFulfilled=120001), deliver=[ContractDelivery(tradeSymbol='IRON_ORE', destination='X1-OE-PM', units=500, fulfilled=0)]), accepted=False, fulfilled=False, expiresAt='2022-04-26T11:57:12.233Z')


v2_calls.TOKEN = st.token

v2_calls.list_contracts()
2022-04-19 16:21:00	INFO	v2.calls: list_contracts: {'total': 1, 'page': 1, 'limit': 20}
Out[51]: [Contract(id='cl263cjvy001401s6alc25vab', faction='COMMERCE_REPUBLIC', type='PROCUREMENT', terms=Term(deadline='2022-05-03T11:57:12.235Z', payment=Payment(onAccepted=30000, onFulfilled=120001), deliver=[ContractDelivery(tradeSymbol='IRON_ORE', destination='X1-OE-PM', units=500, fulfilled=0)]), accepted=True, fulfilled=False, expiresAt='2022-04-26T11:57:12.233Z')]

In [53]: v2_calls.my_agent()
2022-04-19 16:21:35	INFO	v2.calls: my_agent: {'data': {'accountId': 'cl263cjpz000301s6h1e5uxju', 'symbol': 'UMUERO', 'headquarters': 'X1-OE-PM', 'credits': 130000}}
Out[53]: Agent(accountId='cl263cjpz000301s6h1e5uxju', symbol='UMUERO', headquarters='X1-OE-PM', credits=130000)

In [54]: v2_calls.my_account()
2022-04-19 16:21:41	INFO	v2.calls: my_account: {'data': {'account': {'id': 'cl263cjpz000301s6h1e5uxju', 'email': 'temp-account-2e6c1483-275b-4008-a416-77f03a92995d@spacetraders.io', 'discordHandle': None, 'patreonId': None, 'createdAt': '2022-04-19T11:57:12.023Z'}}}
Out[54]:
{'data': {'account': {'id': 'cl263cjpz000301s6h1e5uxju',
   'email': 'temp-account-2e6c1483-275b-4008-a416-77f03a92995d@spacetraders.io',
   'discordHandle': None,
   'patreonId': None,
   'createdAt': '2022-04-19T11:57:12.023Z'}}}


In [55]: v2_calls.view_exports('IRON_ORE')
2022-04-19 16:22:23	INFO	v2.calls: view_exports: {'total': 0, 'page': 1, 'limit': 20}
Out[55]: []

In [56]: v2_calls.view_imports('IRON_ORE')
2022-04-19 16:22:34	INFO	v2.calls: view_imports: {'total': 1, 'page': 1, 'limit': 20}
Out[56]: [MarketTrade(waypointSymbol='X1-OE-PM01', tradeSymbol='IRON_ORE', price=388, tariff=0)]


In [85]: v2_calls.list_markets('X1-OE')
2022-04-19 16:34:22	INFO	v2.calls: list_markets: {'total': 3, 'page': 1, 'limit': 20}
Out[85]: ['X1-OE-PM', 'X1-OE-PM01', 'X1-OE-PM02']

In [20]: v2.navigate_ship(st.ships[0].symbol, 'X1-OE-PM01')
2022-04-19 21:15:06	INFO	v2c: navigate_ship {'fuelCost': 0, 'navigation': {'shipSymbol': 'UMUERO-1', 'departure': 'X1-OE-PM', 'destination': 'X1-OE-PM01', 'durationRemaining': 0, 'arrivedAt': '2022-04-19T18:15:06.989Z'}}
Out[20]: ShipNavigation(shipSymbol='UMUERO-1', departure='X1-OE-PM', destination='X1-OE-PM01', durationRemaining=0, arrivedAt='2022-04-19T18:15:06.989Z')

In [32]: v2.navigate_ship(st.ships[0].symbol, 'X1-OE-A005')
2022-04-19 21:20:26	INFO	v2c: navigate_ship {'fuelCost': 28, 'navigation': {'shipSymbol': 'UMUERO-1', 'departure': 'X1-OE-PM02', 'destination': 'X1-OE-A005', 'durationRemaining': 55, 'arrivedAt': None}}
Out[32]: ShipNavigation(shipSymbol='UMUERO-1', departure='X1-OE-PM02', destination='X1-OE-A005', durationRemaining=55, arrivedAt=None)

In [28]: v2.view_market('X1-OE', 'X1-OE-PM01')
Out[28]: MarketListing(exports=[MarketTrade(waypointSymbol='X1-OE-PM01', tradeSymbol='IRON', price=331, tariff=0), MarketTrade(waypointSymbol='X1-OE-PM01', tradeSymbol='ALUMINUM', price=371, tariff=0), MarketTrade(waypointSymbol='X1-OE-PM01', tradeSymbol='COPPER', price=373, tariff=0)], imports=[MarketTrade(waypointSymbol='X1-OE-PM01', tradeSymbol='IRON_ORE', price=386, tariff=0), MarketTrade(waypointSymbol='X1-OE-PM01', tradeSymbol='ALUMINUM_ORE', price=314, tariff=0), MarketTrade(waypointSymbol='X1-OE-PM01', tradeSymbol='COPPER_ORE', price=329, tariff=0)], exchange=[])

In [29]: v2.view_market('X1-OE', 'X1-OE-PM02')
Out[29]: MarketListing(exports=[MarketTrade(waypointSymbol='X1-OE-PM02', tradeSymbol='COMM_RELAY_I', price=28376, tariff=0)], imports=[MarketTrade(waypointSymbol='X1-OE-PM02', tradeSymbol='ALUMINUM', price=388, tariff=0), MarketTrade(waypointSymbol='X1-OE-PM02', tradeSymbol='ELECTRONICS', price=785, tariff=0)], exchange=[])


In [37]: v2.survey_waypoint(st.ships[0].symbol, None)
2022-04-19 21:26:16	INFO	v2c: survey_waypoint Cooldown(duration=899, expiration='2022-04-19T18:41:16.694Z')
Out[37]:
[Survey(signature='X1-OE-70790D', deposits=['ALUMINUM_ORE', 'COPPER_ORE', 'QUARTZ', 'SILICON'], expiration='2022-04-19T18:36:39.697Z'),
 Survey(signature='X1-OE-43FEC9', deposits=['COPPER_ORE', 'IRON_ORE'], expiration='2022-04-19T18:52:55.697Z'),
 Survey(signature='X1-OE-F46A3B', deposits=['COPPER_ORE', 'QUARTZ'], expiration='2022-04-19T18:50:59.697Z'),
 Survey(signature='X1-OE-10CCC4', deposits=['QUARTZ'], expiration='2022-04-19T18:43:39.697Z'),
 Survey(signature='X1-OE-D62E44', deposits=['ALUMINUM_ORE'], expiration='2022-04-19T18:50:58.697Z'),
 Survey(signature='X1-OE-1803C6', deposits=['ALUMINUM_ORE', 'SILICON'], expiration='2022-04-19T18:49:00.698Z'),
 Survey(signature='X1-OE-C0F1EC', deposits=['ALUMINUM_ORE', 'ALUMINUM_ORE', 'COPPER_ORE', 'QUARTZ'], expiration='2022-04-19T18:33:22.698Z'),
 Survey(signature='X1-OE-4EBBD4', deposits=['ALUMINUM_ORE', 'COPPER_ORE', 'COPPER_ORE'], expiration='2022-04-19T18:32:08.698Z'),
 Survey(signature='X1-OE-2D7A68', deposits=['COPPER_ORE'], expiration='2022-04-19T18:43:08.698Z')]




In [6]: v2.survey_waypoint(st.ships[0].symbol, None)
2022-04-19 22:47:04	INFO	v2c: survey_waypoint Cooldown(duration=899, expiration='2022-04-19T20:02:04.611Z')
Out[6]:
[Survey(signature='X1-OE-42BFF9', deposits=['ALUMINUM_ORE', 'SILICON'], expiration='2022-04-19T19:56:55.614Z'),
 Survey(signature='X1-OE-532712', deposits=['ALUMINUM_ORE', 'ALUMINUM_ORE', 'QUARTZ'], expiration='2022-04-19T20:04:38.615Z'),
 Survey(signature='X1-OE-10A4D0', deposits=['ALUMINUM_ORE', 'ALUMINUM_ORE', 'IRON_ORE', 'QUARTZ'], expiration='2022-04-19T19:54:35.615Z'),
 Survey(signature='X1-OE-EDED67', deposits=['IRON_ORE', 'QUARTZ', 'SILICON', 'SILICON'], expiration='2022-04-19T20:14:14.615Z'),
 Survey(signature='X1-OE-CD9821', deposits=['IRON_ORE'], expiration='2022-04-19T19:59:28.615Z'),
 Survey(signature='X1-OE-C1D21E', deposits=['COPPER_ORE', 'QUARTZ', 'QUARTZ', 'SILICON'], expiration='2022-04-19T19:51:27.616Z')]



In [8]: v2.extract_resources(st.ships[0].symbol, asdict(_6[4]))
2022-04-19 22:47:32	INFO	v2c: extract_resources Cooldown(duration=119, expiration='2022-04-19T19:49:33.046Z')
Out[8]: Extraction(shipSymbol='UMUERO-1', yields=Good(tradeSymbol='IRON_ORE', units=20))


In [11]: v2.ship_dock('UMUERO-1')
2022-04-19 23:04:18	INFO	v2c: dock_ship {'data': {'status': 'DOCKED'}}
Out[11]: 'DOCKED'

In [12]: v2.sell_cargo('UMUERO-1', 'SILICON', 26)
Out[12]: Trade(waypointSymbol='X1-OE-PM', tradeSymbol='SILICON', credits=3692, units=-26)

In [15]: v2.sell_cargo('UMUERO-1', 'QUARTZ', 24)
Out[15]: Trade(waypointSymbol='X1-OE-PM', tradeSymbol='QUARTZ', credits=6816, units=-24)

In [16]: st.ships = v2.ship_list()
2022-04-19 23:05:01	INFO	v2c: list_ships: {'total': 2, 'page': 1, 'limit': 100}

In [17]: st.agent = v2.my_agent()
2022-04-19 23:05:05	INFO	v2c: my_agent: {'data': {'accountId': 'cl263cjpz000301s6h1e5uxju', 'symbol': 'UMUERO', 'headquarters': 'X1-OE-PM', 'credits': 82693}}


"""
