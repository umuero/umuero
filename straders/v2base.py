import json
import logging
from dataclasses import dataclass
from typing import Optional, Union

import ratelimit
import requests
from dacite import from_dict

logger = logging.getLogger("v2c")
logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


@dataclass
class Agent:
    accountId: str
    symbol: str
    headquarters: str
    credits: int


@dataclass
class Faction:
    symbol: str
    name: str
    description: str
    headquarters: str
    traits: list[str]


@dataclass
class Payment:
    onAccepted: int
    onFulfilled: int


@dataclass
class ContractDelivery:
    tradeSymbol: str
    destination: str
    units: int
    fulfilled: int


@dataclass
class Term:
    deadline: str  # "2022-03-09T05:16:59.112Z"
    payment: Payment
    deliver: list[ContractDelivery]


@dataclass
class Contract:
    id: str
    faction: str
    type: str
    terms: Term
    accepted: bool
    fulfilled: bool
    expiresAt: str  # "2022-03-09T05:16:59.112Z"


@dataclass
class Survey:
    signature: str
    deposits: list[str]
    expiration: str  # "2022-03-09T05:16:59.112Z"


@dataclass
class Registration:
    factionSymbol: str
    agentSymbol: str
    fee: int
    role: str


@dataclass
class Integrity:
    frame: int
    reactor: int
    engine: int


@dataclass
class Good:
    tradeSymbol: str
    units: int


@dataclass
class ShipStats:
    fuelTank: int
    cargoLimit: int
    jumpRange: int


@dataclass
class Cooldown:
    duration: int
    expiration: str


@dataclass
class Extraction:
    shipSymbol: str
    yields: Good


@dataclass
class Ship:
    symbol: str
    frame: str
    reactor: str
    engine: str
    fuel: int
    engine: str
    modules: list[str]
    mounts: list[str]
    registration: Registration
    integrity: Integrity
    stats: Optional[ShipStats]
    status: str
    location: Optional[str]
    cargo: list[Good]


@dataclass
class Jump:
    shipSymbol: str
    destination: str


@dataclass
class ShipNavigation:
    shipSymbol: str
    departure: str
    destination: str
    durationRemaining: int
    arrivedAt: Optional[str]


@dataclass
class MarketTrade:
    waypointSymbol: str
    tradeSymbol: str
    price: int
    tariff: int


@dataclass
class Trade:
    waypointSymbol: str
    tradeSymbol: str
    credits: int
    units: int


@dataclass
class MarketListing:
    exports: list[MarketTrade]
    imports: list[MarketTrade]
    exchange: list[MarketTrade]


@dataclass
class System:
    symbol: str
    sector: str
    type: str
    x: int
    y: int
    waypoints: list[str]
    factions: list[Optional[str]]
    charted: bool
    chartedBy: Optional[str]


@dataclass
class Waypoint:
    system: str
    symbol: str
    type: str
    x: int
    y: int
    orbitals: list[str]
    faction: str
    features: list[str]
    traits: list[str]
    charted: bool
    chartedBy: Optional[str]


@dataclass
class Shipyard:
    symbol: str
    faction: str


@dataclass
class ShipyardListing:
    id: str
    waypoint: str
    price: int
    role: str
    frame: str
    reactor: str
    engine: str
    engine: str
    modules: list[str]
    mounts: list[str]


@dataclass
class Meta:
    total: int
    page: int
    limit: int


@dataclass
class State:
    token: str
    agent: Agent
    faction: Faction
    contracts: list[Contract]
    ships: list[Ship]
    systems: dict[str, System]
    markets: dict[str, Optional[MarketListing]]
    availableShips: list[ShipyardListing]
    # 'X1-OE-PM:X1-OE-PM01': 0 # fuel || time
    # 'X1-OE-PM02:X1-OE-A005': 28 fuel, 55 time # fuel || time
    distances: dict[str, int]
    # last market update @ X1-OE-PM: epoch ?? (hersey isoformat time olsun mu ??)
    updates: dict[str, int]
    cooldowns: list[ShipNavigation]  # cooldownlari da buna cevireyim, gezip kontrol edem


class V2:
    host = "https://v2-0-0.alpha.spacetraders.io/"
    token = ""
    timeout = 60
    session: requests.Session = None

    def __init__(self, token: str):
        self.token = token
        self.session = requests.session()
        self.session.headers.update({"Authorization": "Bearer " + self.token})

    @ratelimit.sleep_and_retry
    @ratelimit.limits(calls=2, period=1)
    def callApi(self, method: str, url: str, data={}, noToken=False):
        # Each endpoint also has an example response you can retrieve by sending stub=true as a query parameter or in your POST body.
        # You can also view a stubbed response for an error code by sending errorCode={code} with the stub parameter.
        if method == "GET":
            js = self.session.get(self.host + url, params=data, timeout=self.timeout).json()
        if method == "POST":
            js = self.session.post(self.host + url, json=data, timeout=self.timeout).json()
        if "error" in js:
            logger.info("FAIL: %s %s %s %s", method, url, data, js)
        return js

    ################# LOGIN #################
    def leaderboard(self):
        js = self.callApi("GET", "")
        logger.info(json.dumps(js["stats"]))
        logger.info("mostCredits")
        for node in js.get("leaderboards", {}).get("mostCredits", []):
            logger.info(f"{node['agentSymbol']:>10}: {node['credits']:>12,}")
        logger.info("mostSubmittedCharts")
        for node in js.get("leaderboards", {}).get("mostSubmittedCharts", []):
            logger.info(f"{node['agentSymbol']:>10}: {node['chartCount']:>12,}")

    def agent_register(self, username: str, faction: str = "COMMERCE_REPUBLIC") -> State:
        js = self.callApi("POST", "agents", {"symbol": username, "faction": faction}, noToken=True)
        logger.info("agent_register: %s %s %s", username, faction, js)
        js["data"]["contracts"] = [js["data"]["contract"]]
        js["data"]["ships"] = [js["data"]["ship"]]
        js["data"]["systems"] = {s.symbol: s for s in self.system_list()}
        js["data"]["markets"] = []
        st = from_dict(State, js["data"])
        return st

    def my_agent(self) -> Agent:
        js = self.callApi("GET", "my/agent")
        logger.info("my_agent: %s", js)
        return from_dict(Agent, js["data"])

    def my_account(self):
        js = self.callApi("GET", "my/account")
        logger.info("my_account: %s", js)
        return js

    ################# SYSTEMS #################
    def system_list(self, page: int = 1, limit: int = 200) -> list[System]:
        js = self.callApi("GET", "systems", {"page": page, "limit": limit})
        logger.info("system_list: %s", js["meta"])
        return [from_dict(System, j) for j in js["data"]]

    def system_view(self, system_symbol: str) -> System:
        js = self.callApi("GET", f"systems/{system_symbol}")
        logger.info("system_view: %s", js)
        return from_dict(System, js["data"])

    def waypoint_list(self, system_symbol: str) -> list[Waypoint]:
        js = self.callApi("GET", f"systems/{system_symbol}/waypoints")
        logger.info("waypoint_list: %s", js["meta"])
        return [from_dict(Waypoint, j) for j in js["data"]]

    def waypoint_view(self, system_symbol: str, waypoint_symbol: str) -> Waypoint:
        js = self.callApi("GET", f"systems/{system_symbol}/waypoints/{waypoint_symbol}")
        return from_dict(Waypoint, js["data"])

    ################# SHIPYARD #################
    def shipyard_list(self, system_symbol: str) -> list[Shipyard]:
        js = self.callApi("GET", f"systems/{system_symbol}/shipyards")
        logger.info("shipyard_list: %s", js["meta"])
        return [from_dict(Shipyard, j) for j in js["data"]]

    def shipyard_view(self, system_symbol: str, waypoint_symbol: str) -> Shipyard:
        js = self.callApi("GET", f"systems/{system_symbol}/shipyards/{waypoint_symbol}")
        return from_dict(Shipyard, js["data"])

    def shipyard_listing(self, system_symbol: str, waypoint_symbol: str) -> list[ShipyardListing]:
        js = self.callApi("GET", f"systems/{system_symbol}/shipyards/{waypoint_symbol}/ships")
        logger.info("shipyard_listing: %s", js.get("meta"))
        return [from_dict(ShipyardListing, j) for j in js["data"]]

    def shipyard_buy(self, id: str) -> list[str]:
        js = self.callApi("POST", f"my/ships", {"id": id})
        logger.info("shipyard_buy: %d", js["data"]["credits"])
        return from_dict(Ship, js["data"]["ship"])

    ################# MARKET #################
    def market_list(self, system_symbol: str) -> list[str]:
        js = self.callApi("GET", f"systems/{system_symbol}/markets")
        logger.info("market_list: %s", js["meta"])
        return js["data"]

    def market_view(self, system_symbol: str, waypoint_symbol: str) -> MarketListing:
        js = self.callApi("GET", f"systems/{system_symbol}/markets/{waypoint_symbol}")
        return from_dict(MarketListing, js["data"])

    def market_imports(self, trade_symbol: str) -> list[MarketTrade]:
        js = self.callApi("GET", f"trade/{trade_symbol}/imports")
        logger.info("market_imports: %s", js["meta"])
        return [from_dict(MarketTrade, j) for j in js["data"]]

    def market_exports(self, trade_symbol: str) -> list[MarketTrade]:
        js = self.callApi("GET", f"trade/{trade_symbol}/exports")
        logger.info("market_exports: %s", js["meta"])
        return [from_dict(MarketTrade, j) for j in js["data"]]

    def market_exchange(self, trade_symbol: str) -> list[MarketTrade]:
        js = self.callApi("GET", f"trade/{trade_symbol}/exchange")
        logger.info("market_exchange: %s", js["meta"])
        return [from_dict(MarketTrade, j) for j in js["data"]]

    ################# CONTRACT #################
    def contract_list(self) -> list[Contract]:
        js = self.callApi("GET", "my/contracts")
        logger.info("list_contracts: %s", js["meta"])
        return [from_dict(Contract, j) for j in js["data"]]

    def contract_view(self, contract_id) -> Contract:
        js = self.callApi("GET", f"my/contracts/{contract_id}")
        return from_dict(Contract, js["data"])

    def contract_accept(self, contract_id) -> Contract:
        js = self.callApi("POST", f"my/contracts/{contract_id}/accept")
        return from_dict(Contract, js["data"])

    ################# SHIP #################
    def ship_list(self) -> list[Ship]:
        js = self.callApi("GET", f"my/ships", {"limit": 100})
        logger.info("list_ships: %s", js["meta"])
        return [from_dict(Ship, j) for j in js["data"]]

    def ship_view(self, ship_symbol: str) -> Ship:
        js = self.callApi("POST", f"my/ships/{ship_symbol}")
        return from_dict(Ship, js["data"])

    def ship_cooldown_scan(self, ship_symbol: str) -> Cooldown:
        js = self.callApi("GET", f"my/ships/{ship_symbol}/scan")
        return from_dict(Cooldown, js["data"]["cooldown"])

    def ship_cooldown_extract(self, ship_symbol: str) -> Cooldown:
        js = self.callApi("GET", f"my/ships/{ship_symbol}/extract")
        return from_dict(Cooldown, js["data"]["cooldown"])

    def ship_cooldown_survey(self, ship_symbol: str) -> Cooldown:
        js = self.callApi("GET", f"my/ships/{ship_symbol}/survey")
        return from_dict(Cooldown, js["data"]["cooldown"])

    def ship_cooldown_jump(self, ship_symbol: str) -> Cooldown:
        js = self.callApi("GET", f"my/ships/{ship_symbol}/jump")
        return from_dict(Cooldown, js["data"]["cooldown"])

    def ship_status_navigation(self, ship_symbol: str) -> ShipNavigation:
        js = self.callApi("GET", f"my/ships/{ship_symbol}/navigate")
        return from_dict(ShipNavigation, js["data"]["navigation"])

    ################# NAVIGATION #################
    def ship_refuel(self, ship_symbol: str):
        js = self.callApi("POST", f"my/ships/{ship_symbol}/refuel")
        logger.info("refuel_ship: %s", js)
        return js["data"]

    def ship_dock(self, ship_symbol: str) -> str:
        # mode=STEALTH
        js = self.callApi("POST", f"my/ships/{ship_symbol}/dock")
        logger.info("dock_ship %s", js)
        return js["data"]["status"]

    def ship_orbit(self, ship_symbol: str) -> str:
        # mode=BLOCKADE,SPY,PATROL
        js = self.callApi("POST", f"my/ships/{ship_symbol}/orbit")
        logger.info("orbit_ship %s", js)
        return js["data"]["status"]

    def ship_jump(self, ship_symbol: str, destination: str) -> Jump:
        js = self.callApi("POST", f"my/ships/{ship_symbol}/jump", {"destination": destination})
        cooldown = from_dict(Cooldown, js["data"]["cooldown"])
        logger.info("jump_ship %s", cooldown)
        return from_dict(Jump, js["data"]["jump"])

    def ship_navigate(self, ship_symbol: str, destination: str) -> ShipNavigation:
        # mode=CRUISE ...
        js = self.callApi("POST", f"my/ships/{ship_symbol}/navigate", {"destination": destination})
        logger.info("navigate_ship %s", js["data"])
        return from_dict(ShipNavigation, js["data"]["navigation"])

    ################# SHIP ACTIONS #################
    def chart_waypoint(self, ship_symbol: str) -> list[str]:
        js = self.callApi("POST", f"my/ships/{ship_symbol}/chart")
        logger.info("chart_waypoint: %s", js)
        return js["data"]

    def deploy_asset(self, ship_symbol: str) -> str:
        js = self.callApi("POST", f"my/ships/{ship_symbol}/deploy")
        logger.info("deploy_asset: %s", js)
        return js

    def activate_scan(self, ship_symbol: str, mode: str) -> Union[Waypoint, System, list[Ship]]:
        # mode: APPROACHING_SHIPS DEPARTING_SHIPS SYSTEM WAYPOINT
        js = self.callApi("POST", f"my/ships/{ship_symbol}/scan", {"mode": mode})
        cooldown = from_dict(Cooldown, js["data"]["cooldown"])
        logger.info("activate_scan %s", cooldown)
        if "waypoint" in js["data"]:
            return from_dict(Waypoint, js["data"]["waypoint"])
        if "system" in js["data"]:
            return from_dict(System, js["data"]["system"])
        if "ships" in js["data"]:
            return [from_dict(Cooldown, j) for j in js["data"]["ships"]]

    def extract_resources(self, ship_symbol: str, survey: Optional[Survey]) -> Extraction:
        if survey:
            js = self.callApi("POST", f"my/ships/{ship_symbol}/extract", {"survey": survey})
        else:
            js = self.callApi("POST", f"my/ships/{ship_symbol}/extract")  # random extraction
        cooldown = from_dict(Cooldown, js["data"]["cooldown"])
        logger.info("extract_resources %s", cooldown)
        js["data"]["extraction"]["yields"] = js["data"]["extraction"]["yield"]
        return from_dict(Extraction, js["data"]["extraction"])

    def survey_waypoint(self, ship_symbol: str, survey: Optional[Survey]) -> list[Survey]:
        js = self.callApi("POST", f"my/ships/{ship_symbol}/survey", {"survey": survey})
        cooldown = from_dict(Cooldown, js["data"]["cooldown"])
        logger.info("survey_waypoint %s", cooldown)
        return [from_dict(Survey, j) for j in js["data"]["surveys"]]

    ################# SHIP CARGO ACTIONS #################
    def jettison_cargo(self, ship_symbol: str, trade_symbol: str, units: int) -> Good:
        js = self.callApi("POST", f"my/ships/{ship_symbol}/jettison", {"tradeSymbol": trade_symbol, "units": units})
        return from_dict(Good, js["data"])

    def contract_deliver(self, contract_id: str, ship_symbol: str, trade_symbol: str, units: int) -> ContractDelivery:
        js = self.callApi(
            "POST", f"my/ships/{ship_symbol}/deliver", {"contractId": contract_id, "tradeSymbol": trade_symbol, "units": units}
        )
        logger.info("deliver_contract %s", js)
        return from_dict(ContractDelivery, js["data"])

    def purchase_cargo(self, ship_symbol: str, trade_symbol: str, units: int) -> Trade:
        js = self.callApi("POST", f"my/ships/{ship_symbol}/purchase", {"tradeSymbol": trade_symbol, "units": units})
        return from_dict(Trade, js["data"])

    def sell_cargo(self, ship_symbol: str, trade_symbol: str, units: int) -> Trade:
        js = self.callApi("POST", f"my/ships/{ship_symbol}/sell", {"tradeSymbol": trade_symbol, "units": units})
        return from_dict(Trade, js["data"])
