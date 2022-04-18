from dataclasses import dataclass
from typing import Optional, Union


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
    frame: str
    reactor: str
    engine: str


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
    location: str
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
    expexchangeorts: list[MarketTrade]


@dataclass
class System:
    symbol: str
    sector: str
    type: str
    x: int
    y: int
    waypoints: list[str]
    factions: list[str]
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
class State:
    token: str
    agent: Agent
    faction: Faction
    contracts: list[Contract]
    ships: list[Ship]
    systems: list[System]
    markets: list[MarketListing]


@dataclass
class Meta:
    total: int
    page: int
    limit: int
