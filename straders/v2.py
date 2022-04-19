import argparse
import ast
import datetime
import logging
import json
import time
import math
from dataclasses import asdict
from dacite import from_dict
from regex import P
from v2base import *


USERNAME = "umuero"
TOKEN = ""
SAVE = "dataV2.json"
MARKET_REFRESH_NEEDED = 60 * 60 * 2  # 2h

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


def locToSys(location: str) -> str:
    return "-".join(location.split("-")[:2])


def cooldown2shipNav(ship: Ship, cooldownType: str, cooldown: Cooldown) -> ShipNavigation:
    return ShipNavigation(ship.symbol, cooldownType, "", cooldown.duration, cooldown.expiration)


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
            currentEpoch = int(time.time())
            currentIso = datetime.datetime.utcnow().isoformat("T", "milliseconds") + "Z"
            st.agent = v2.my_agent()
            st.contracts = v2.contract_list()
            st.ships = v2.ship_list()
            if args.extra:  # or loopCtr % 60 == 0:
                # daha seyrek cekmek lazim
                st.systems = {s.symbol: s for s in v2.system_list()}
                st.availableShips = []
                for system in st.systems.values():
                    if system.charted:
                        for waypoint in v2.waypoint_list(system.symbol):
                            if waypoint.symbol not in st.waypoints:
                                st.waypoints[waypoint.symbol] = waypoint
                            if "MARKETPLACE" in waypoint.features:
                                if waypoint.symbol not in st.markets:
                                    st.markets[waypoint.symbol] = None
                            if "SHIPYARD" in waypoint.features:
                                try:
                                    st.availableShips.extend(v2.shipyard_listing(system.symbol, waypoint.symbol))
                                except Exception:
                                    logger.info("not charted waypoint ??")

            st.cooldowns = [cd for cd in st.cooldowns if cd.arrivedAt is not None and cd.arrivedAt > currentIso]
            orders = {}
            for ship in st.ships:
                if ship.location is None:
                    continue
                shipSys = locToSys(ship.location)

                if ship.location in st.markets:
                    v2.ship_dock(ship.symbol)
                    st.markets[ship.location] = v2.market_view(shipSys, ship.location)
                    st.updates[ship.location] = currentEpoch
                    importItems = set([mt.tradeSymbol for mt in st.markets[ship.location].imports])
                    for good in ship.cargo:
                        if good.tradeSymbol in importItems:
                            v2.cargo_sell(ship.symbol, good.tradeSymbol, good.units)
                    v2.ship_refuel(ship.symbol)
                    v2.ship_orbit(ship.symbol)

                if ship.registration.role == "COMMAND":
                    for marketNode in st.markets.keys():
                        if not marketNode.startswith(shipSys):
                            continue
                        if currentEpoch - st.updates.get(marketNode, 0) > MARKET_REFRESH_NEEDED:
                            logger.info(
                                f"{ship.symbol} to update market: {marketNode} @{currentEpoch - st.updates.get(marketNode, 0)}"
                            )
                            st.cooldowns.append(v2.ship_navigate(ship.symbol, marketNode))
                            orders[ship.symbol] = "marketUpdate"
                            break
                    if ship.symbol in orders:
                        continue
                cargoSum = sum([g.units for g in ship.cargo])
                if st.waypoints[ship.location].type == "ASTEROID_FIELD":
                    if cargoSum < ship.stats.cargoLimit:
                        cd = [cd for cd in st.cooldowns if cd.shipSymbol == ship.symbol and cd.departure == "extract"]
                        if cd:
                            logger.info(f"{ship.symbol} waiting cd {cd[0].arrivedAt}")
                            continue
                        cooldown, extraction = v2.extract_resources(ship.symbol)
                        logger.info(f"{ship.symbol} extracted {extraction.yields.units} {extraction.yields.tradeSymbol}")
                        st.cooldowns.append(cooldown2shipNav(ship, "extract", cooldown))
                        continue
                else:
                    if cargoSum == 0:
                        astro = [w for w in st.waypoints.values() if w.system == shipSys and w.type == "ASTEROID_FIELD"]
                        if astro:
                            logger.info(f"{ship.symbol} to going to mining: {astro[0].symbol}")
                            st.cooldowns.append(v2.ship_navigate(ship.symbol, astro[0].symbol))
                            continue
                    else:
                        # market market gezeelim
                        for good in ship.cargo:
                            marketT = v2.market_imports(good.tradeSymbol)
                            if marketT and locToSys(marketT[0].waypointSymbol) == shipSys:
                                logger.info(f"{ship.symbol} to going to {marketT[0].waypointSymbol} to sell: {good.tradeSymbol}")
                                st.cooldowns.append(v2.ship_navigate(ship.symbol, marketT[0].waypointSymbol))
                                orders[ship.symbol] = "goingMarket"
                                break
                        if ship.symbol in orders:
                            continue

            # if role == command ?? -> none butun marketleri gez, save et
            # if role == solar && extractor -> OE'de extract et, sat

            # astroid_field e git; None survey et :P
            # kargo full ise - git dock et sat, refuel;

            # contract delivery icin orbit kafi,
            # market icin dock olacak
            # if in marketNode --> refuel (@dock)

            # X1-OE de gez, survey extract -> sat ??
            # her navigate result -> st.cooldowns.append
            # her survey extract scan result -> st.cooldowns.append

            saveJson(st)
        except Exception as e:
            logger.exception("main loop failed")

        loopCtr += 1
        print("sleeping", args.sleep)
        if args.extra:
            break
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

In [21]: v2.ship_dock('UMUERO-1')
2022-04-19 23:19:41	INFO	v2c: dock_ship {'data': {'status': 'DOCKED'}}
Out[21]: 'DOCKED'

In [22]: v2.ship_refuel('UMUERO-1')
2022-04-19 23:19:42	INFO	v2c: refuel_ship: {'data': {'credits': -240, 'fuel': 56}}
{'credits': -240, 'fuel': 100}

Out[22]: {'credits': -240, 'fuel': 56}



In [40]: v2.survey_waypoint('UMUERO-2', 'X1-OE-25X')
2022-04-19 23:23:31	INFO	v2c: FAIL: POST my/ships/UMUERO-2/survey {'survey': 'X1-OE-25X'} {'error': {'message': 'Ship survey failed. Ship UMUERO-2 is not at a valid location, such as an asteroid field.', 'code': 4225}}


[System(symbol='X1-OE', sector='X1', type='RED_STAR', x=0, y=0, waypoints=['X1-OE-25X', 'X1-OE-A005', 'X1-OE-PM', 'X1-OE-PM01', 'X1-OE-PM02'], factions=['SPACERS_GUILD', 'MINERS_COLLECTIVE', 'COMMERCE_REPUBLIC'], charted=True, chartedBy=None),
 System(symbol='X1-EV', sector='X1', type='ORANGE_STAR', x=2, y=3, waypoints=['X1-EV-A004'], factions=['COMMERCE_REPUBLIC'], charted=True, chartedBy=None),
 System(symbol='X1-ZZ', sector='X1', type='BLUE_STAR', x=-5, y=11, waypoints=['X1-ZZ-7', 'X1-ZZ-7-EE'], factions=['ZANZIBAR_TRIKES'], charted=True, chartedBy='EMBER-1'),
 System(symbol='X1-SJ5', sector='X1', type='YOUNG_STAR', x=-15, y=3, waypoints=['X1-SJ5-89481F', 'X1-SJ5-92302B', 'X1-SJ5-22350A'], factions=[None], charted=True, chartedBy='VIRIDIS-1'),
 System(symbol='X1-SM8', sector='X1', type='BLUE_STAR', x=-13, y=-12, waypoints=['X1-SM8-29141A', 'X1-SM8-02072A', 'X1-SM8-13033C', 'X1-SM8-75624X', 'X1-SM8-33005A', 'X1-SM8-28590Z'], factions=[None], charted=True, chartedBy='VIRIDIS-1')]


 [Waypoint(system='X1-OE', symbol='X1-OE-PM', type='PLANET', x=2, y=10, orbitals=['X1-OE-PM01', 'X1-OE-PM02'], faction='COMMERCE_REPUBLIC', features=['MARKETPLACE', 'SHIPYARD'], traits=['OVERCROWDED', 'HIGH_TECH', 'BUREAUCRATIC', 'TRADING_HUB', 'TEMPERATE', 'COMM_RELAY_I'], charted=True, chartedBy=None),
 Waypoint(system='X1-OE', symbol='X1-OE-PM01', type='MOON', x=2, y=10, orbitals=[], faction='COMMERCE_REPUBLIC', features=['MARKETPLACE'], traits=['WEAK_GRAVITY', 'COMM_RELAY_I'], charted=True, chartedBy=None),
 Waypoint(system='X1-OE', symbol='X1-OE-A005', type='ASTEROID_FIELD', x=-26, y=15, orbitals=[], faction='MINERS_COLLECTIVE', features=[], traits=['COMMON_METAL_DEPOSITS'], charted=True, chartedBy=None),
 Waypoint(system='X1-OE', symbol='X1-OE-25X', type='JUMP_GATE', x=-5, y=60, orbitals=[], faction='SPACERS_GUILD', features=[], traits=[], charted=True, chartedBy=None),
 Waypoint(system='X1-OE', symbol='X1-OE-PM02', type='MOON', x=2, y=10, orbitals=[], faction='COMMERCE_REPUBLIC', features=['MARKETPLACE'], traits=['WEAK_GRAVITY', 'COMM_RELAY_I'], charted=True, chartedBy=None)]


 [Waypoint(system='X1-EV', symbol='X1-EV-A004', type='PLANET', x=-8, y=5, orbitals=[], faction='COMMERCE_REPUBLIC', features=['MARKETPLACE', 'SHIPYARD'], traits=['SPRAWLING_CITIES', 'INDUSTRIAL', 'SALT_FLATS', 'CANYONS', 'SCARCE_LIFE', 'BREATHABLE_ATMOSPHERE', 'ROCKY', 'COMM_RELAY_I'], charted=True, chartedBy=None)]


 [Waypoint(system='X1-ZZ', symbol='X1-ZZ-7', type='GAS_GIANT', x=-30, y=85, orbitals=['X1-ZZ-7-EE'], faction='ZANZIBAR_TRIKES', features=[], traits=['CORROSIVE_ATMOSPHERE', 'STRONG_GRAVITY', 'VIBRANT_AURORAS', 'COMM_RELAY_I'], charted=True, chartedBy='EMBER-1'),
 Waypoint(system='X1-ZZ', symbol='X1-ZZ-7-EE', type='ORBITAL_STATION', x=-30, y=85, orbitals=[], faction='UNCHARTED', features=['UNCHARTED'], traits=['UNCHARTED'], charted=False, chartedBy=None)]


 [Waypoint(system='X1-SJ5', symbol='X1-SJ5-22350A', type='PLANET', x=-9, y=10, orbitals=['X1-SJ5-89481F'], faction=None, features=['MARKETPLACE'], traits=['VOLCANIC', 'SCATTERED_SETTLEMENTS'], charted=True, chartedBy='VIRIDIS-1'),
 Waypoint(system='X1-SJ5', symbol='X1-SJ5-89481F', type='MOON', x=-9, y=10, orbitals=[], faction='UNCHARTED', features=['UNCHARTED'], traits=['UNCHARTED'], charted=False, chartedBy=None),
 Waypoint(system='X1-SJ5', symbol='X1-SJ5-92302B', type='PLANET', x=37, y=-15, orbitals=[], faction='UNCHARTED', features=['UNCHARTED'], traits=['UNCHARTED'], charted=False, chartedBy=None)]


[Waypoint(system='X1-SM8', symbol='X1-SM8-28590Z', type='PLANET', x=-18, y=-2, orbitals=['X1-SM8-29141A'], faction=None, features=['MARKETPLACE'], traits=['SWAMP', 'SPRAWLING_CITIES', 'BLACK_MARKET', 'BUREAUCRATIC', 'COMM_RELAY_I'], charted=True, chartedBy='VIRIDIS-1'),
 Waypoint(system='X1-SM8', symbol='X1-SM8-29141A', type='MOON', x=-18, y=-2, orbitals=[], faction='UNCHARTED', features=['UNCHARTED'], traits=['UNCHARTED'], charted=False, chartedBy=None),
 Waypoint(system='X1-SM8', symbol='X1-SM8-02072A', type='PLANET', x=-2, y=-40, orbitals=['X1-SM8-13033C', 'X1-SM8-75624X'], faction='UNCHARTED', features=['UNCHARTED'], traits=['UNCHARTED'], charted=False, chartedBy=None),
 Waypoint(system='X1-SM8', symbol='X1-SM8-13033C', type='MOON', x=-2, y=-40, orbitals=[], faction='UNCHARTED', features=['UNCHARTED'], traits=['UNCHARTED'], charted=False, chartedBy=None),
 Waypoint(system='X1-SM8', symbol='X1-SM8-75624X', type='MOON', x=-2, y=-40, orbitals=[], faction='UNCHARTED', features=['UNCHARTED'], traits=['UNCHARTED'], charted=False, chartedBy=None),
 Waypoint(system='X1-SM8', symbol='X1-SM8-33005A', type='PLANET', x=-8, y=59, orbitals=[], faction='UNCHARTED', features=['UNCHARTED'], traits=['UNCHARTED'], charted=False, chartedBy=None)]
"""
