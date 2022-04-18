import requests
import ratelimit
import logging
from dacite import from_dict
from v2_types import *

HOST = "https://v2-0-0.alpha.spacetraders.io/"
TOKEN = ""
TIMEOUT = 60

logger = logging.getLogger("v2_calls")

session = requests.session()
session.headers.update()


@ratelimit.sleep_and_retry
@ratelimit.limits(calls=2, period=1)
def callApi(method: str, url: str, data={}, noToken=False):
    # Each endpoint also has an example response you can retrieve by sending stub=true as a query parameter or in your POST body.
    # You can also view a stubbed response for an error code by sending errorCode={code} with the stub parameter.

    # All routes that are prefixed with /my will require an agent access token to be passed as an Authorization: Bearer {token} header in the request.
    headers = {}
    if not noToken:
        headers = {"Authorization": "Bearer " + TOKEN}
    if method == "GET":
        js = requests.get(HOST + url, params=data, headers=headers, timeout=TIMEOUT).json()
    if method == "POST":
        js = requests.post(HOST + url, params=data, headers=headers, timeout=TIMEOUT).json()
    if method == "PUT":
        js = requests.put(HOST + url, params=data, headers=headers, timeout=TIMEOUT).json()
    if method == "DELETE":
        js = requests.delete(HOST + url, params=data, headers=headers, timeout=TIMEOUT).json()
    if "error" in js:
        logger.info("FAIL: %s %s %s %s", method, url, data, js)
    return js


################# LOGIN #################
def register_agent(username: str, faction: str = "COMMERCE_REPUBLIC") -> State:
    js = callApi("POST", "agents", {"symbol": username, "faction": faction}, noToken=True)
    logger.info("register_agent: %s %s %s", username, faction, js)
    st = State()
    st.token = js["data"]["token"]
    st.agent = from_dict(Agent, js["data"]["agent"])
    st.faction = from_dict(Faction, js["data"]["faction"])
    st.contracts = [from_dict(Contract, js["data"]["contract"])]
    st.ships = [from_dict(Contract, js["data"]["ship"])]
    return st


################# SYSTEMS #################
def list_systems() -> list[System]:
    js = callApi("GET", "systems")
    logger.info("list_systems: %s", js["meta"])
    return [from_dict(System, j) for j in js["data"]]


def view_system(system_symbol: str) -> System:
    js = callApi("GET", f"systems/{system_symbol}")
    logger.info("view_system: %s", js)
    return from_dict(System, js["data"])


def list_waypoints(system_symbol: str) -> list[Waypoint]:
    js = callApi("GET", f"systems/{system_symbol}/waypoints")
    logger.info("list_waypoints: %s", js["meta"])
    return [from_dict(Waypoint, j) for j in js["data"]]


def view_waypoint(system_symbol: str, waypoint_symbol: str) -> Waypoint:
    js = callApi("GET", f"systems/{system_symbol}/waypoints/{waypoint_symbol}")
    return from_dict(Waypoint, js["data"])


def list_shipyards(system_symbol: str) -> list[Shipyard]:
    js = callApi("GET", f"systems/{system_symbol}/shipyards")
    logger.info("list_shipyards: %s", js["meta"])
    return [from_dict(Shipyard, j) for j in js["data"]]


def view_shipyard(system_symbol: str, waypoint_symbol: str) -> Shipyard:
    js = callApi("GET", f"systems/{system_symbol}/shipyards/{waypoint_symbol}")
    return from_dict(Shipyard, js["data"])


def list_shipyard_listing(system_symbol: str, waypoint_symbol: str) -> list[ShipyardListing]:
    js = callApi("GET", f"systems/{system_symbol}/shipyards/{waypoint_symbol}/ships")
    logger.info("list_shipyard_listing: %s", js["meta"])
    return [from_dict(ShipyardListing, j) for j in js["data"]]


def list_markets(system_symbol: str) -> list[str]:
    js = callApi("GET", f"systems/{system_symbol}/markets")
    logger.info("list_markets: %s", js["meta"])
    return js["data"]


def view_market(system_symbol: str, waypoint_symbol: str) -> MarketListing:
    js = callApi("GET", f"systems/{system_symbol}/markets/{waypoint_symbol}")
    return from_dict(MarketListing, js["data"])


def view_imports(trade_symbol: str) -> list[MarketTrade]:
    js = callApi("GET", f"trade/{trade_symbol}/imports")
    logger.info("view_imports: %s", js["meta"])
    return [from_dict(MarketTrade, j) for j in js["data"]]


def view_exports(trade_symbol: str) -> list[MarketTrade]:
    js = callApi("GET", f"trade/{trade_symbol}/exports")
    logger.info("view_exports: %s", js["meta"])
    return [from_dict(MarketTrade, j) for j in js["data"]]


def view_exchange(trade_symbol: str) -> list[MarketTrade]:
    js = callApi("GET", f"trade/{trade_symbol}/exchange")
    logger.info("view_exchange: %s", js["meta"])
    return [from_dict(MarketTrade, j) for j in js["data"]]


################# MY #################
def my_agent() -> Agent:
    js = callApi("GET", "my/agent")
    logger.info("my_agent: %s", js)
    return from_dict(Agent, js["data"])


def my_account():
    js = callApi("GET", "my/account")
    logger.info("my_account: %s", js)
    return js


def list_contracts() -> list[Contract]:
    js = callApi("GET", "my/contracts")
    logger.info("list_contracts: %s", js["meta"])
    return [from_dict(Contract, j) for j in js["data"]]


def view_contract(contract_id) -> Contract:
    js = callApi("GET", f"my/contracts/{contract_id}")
    return from_dict(Contract, js["data"])


def accept_contract(contract_id) -> Contract:
    js = callApi("POST", f"my/contracts/{contract_id}/accept")
    return from_dict(Contract, js["data"])


def list_ships() -> list[Ship]:
    js = callApi("POST", f"my/ships")
    logger.info("list_ships: %s", js["meta"])
    return [from_dict(Ship, j) for j in js["data"]]


def view_ship(ship_symbol: str) -> Ship:
    js = callApi("POST", f"my/ships/{ship_symbol}")
    return from_dict(Ship, js["data"])


def view_ship_scan_cooldown(ship_symbol: str) -> Cooldown:
    js = callApi("GET", f"my/ships/{ship_symbol}/scan")
    return from_dict(Cooldown, js["data"]["cooldown"])


def view_ship_survey_cooldown(ship_symbol: str) -> Cooldown:
    js = callApi("GET", f"my/ships/{ship_symbol}/survey")
    return from_dict(Cooldown, js["data"]["cooldown"])


def view_ship_jump_cooldown(ship_symbol: str) -> Cooldown:
    js = callApi("GET", f"my/ships/{ship_symbol}/jump")
    return from_dict(Cooldown, js["data"]["cooldown"])


def view_ship_navigation(ship_symbol: str) -> ShipNavigation:
    js = callApi("GET", f"my/ships/{ship_symbol}/navigate")
    return from_dict(ShipNavigation, js["data"]["navigation"])


def chart_waypoint(ship_symbol: str) -> list[str]:
    js = callApi("POST", f"my/ships/{ship_symbol}/chart")
    logger.info("chart_waypoint: %s", js)
    return js["data"]


def refuel_ship(ship_symbol: str):
    js = callApi("POST", f"my/ships/{ship_symbol}/refuel")
    logger.info("refuel_ship: %s", js)
    return js["data"]


def dock_ship(ship_symbol: str) -> str:
    # mode=STEALTH
    js = callApi("POST", f"my/ships/{ship_symbol}/dock")
    logger.info("dock_ship %s", js)
    return js["data"]["status"]


def orbit_ship(ship_symbol: str) -> str:
    # mode=BLOCKADE,SPY,PATROL
    js = callApi("POST", f"my/ships/{ship_symbol}/orbit")
    logger.info("orbit_ship %s", js)
    return js["data"]["status"]


def jump_ship(ship_symbol: str, destination: str) -> Jump:
    js = callApi("POST", f"my/ships/{ship_symbol}/jump", {"destination": destination})
    cooldown = from_dict(Cooldown, js["data"]["cooldown"])
    logger.info("jump_ship %s", cooldown)
    return from_dict(Jump, js["data"]["jump"])


def navigate_ship(ship_symbol: str, destination: str) -> ShipNavigation:
    # mode=CRUISE ...
    js = callApi("POST", f"my/ships/{ship_symbol}/navigate", {"destination": destination})
    logger.info("navigate_ship %s", js["data"])
    return from_dict(ShipNavigation, js["data"]["navigation"])


def deploy_asset(ship_symbol: str) -> str:
    js = callApi("POST", f"my/ships/{ship_symbol}/deploy")
    logger.info("deploy_asset: %s", js)
    return js


def purchase_ship(id: str) -> list[str]:
    js = callApi("POST", f"my/ships", {id: id})
    logger.info("purchase_ship: %d", js["data"]["credits"])
    return from_dict(Ship, js["data"]["ship"])


def activate_scan(ship_symbol: str, mode: str) -> Union(Waypoint, System, list[Ship]):
    # mode: APPROACHING_SHIPS DEPARTING_SHIPS SYSTEM WAYPOINT
    js = callApi("POST", f"my/ships/{ship_symbol}/scan", {"mode": mode})
    cooldown = from_dict(Cooldown, js["data"]["cooldown"])
    logger.info("activate_scan %s", cooldown)
    if "waypoint" in js["data"]:
        return from_dict(Waypoint, js["data"]["waypoint"])
    if "system" in js["data"]:
        return from_dict(System, js["data"]["system"])
    if "ships" in js["data"]:
        return [from_dict(Cooldown, j) for j in js["data"]["ships"]]


def extract_resources(ship_symbol: str, survey: Optional[Survey]) -> Extraction:
    if survey:
        js = callApi("POST", f"my/ships/{ship_symbol}/extract", {"survey": survey})
    else:
        js = callApi("POST", f"my/ships/{ship_symbol}/extract")  # random extraction
    cooldown = from_dict(Cooldown, js["data"]["cooldown"])
    logger.info("extract_resources %s", cooldown)
    return from_dict(Extraction, js["data"]["extraction"])


def survey_waypoint(ship_symbol: str, survey: Optional[Survey]) -> list[Survey]:
    js = callApi("POST", f"my/ships/{ship_symbol}/survey", {"survey": survey})
    cooldown = from_dict(Cooldown, js["data"]["cooldown"])
    logger.info("survey_waypoint %s", cooldown)
    return [from_dict(Survey, j) for j in js["data"]["surveys"]]


def jettison_cargo(ship_symbol: str, trade_symbol: str, units: int) -> Good:
    js = callApi("POST", f"my/ships/{ship_symbol}/jettison", {"tradeSymbol": trade_symbol, "units": units})
    return from_dict(Good, js["data"])


def deliver_contract(contract_id: str, ship_symbol: str, trade_symbol: str, units: int) -> ContractDelivery:
    js = callApi(
        "POST", f"my/ships/{ship_symbol}/deliver", {"contractId": contract_id, "tradeSymbol": trade_symbol, "units": units}
    )
    logger.info("deliver_contract %s", js)
    return from_dict(ContractDelivery, js["data"])  # ["data"]["data"] ??


def purchase_cargo(ship_symbol: str, trade_symbol: str, units: int) -> Trade:
    js = callApi("POST", f"my/ships/{ship_symbol}/purchase", {"tradeSymbol": trade_symbol, "units": units})
    return from_dict(Trade, js["data"])


def sell_cargo(ship_symbol: str, trade_symbol: str, units: int) -> Trade:
    js = callApi("POST", f"my/ships/{ship_symbol}/sell", {"tradeSymbol": trade_symbol, "units": units})
    return from_dict(Trade, js["data"])
