import requests
import json
import datetime
import time
import math
import random
import argparse
import traceback
from collections import Counter

USERNAME = 'umuero'
TOKEN = ''
HOST = 'https://api.spacetraders.io'  # https://spacetraders.io/
SAVE = 'data.json'
REQUEST_SLEEP = 1


""" TASKS
- limited-trades (fuel-metal ...) ??
- trade fuel hesabindan totalFuel yerine time hesaplasak ??
- structure - structure trades
- structure_OE & structure_ZY1 roles 
- '{:,}'.format(10000)
"""

initialOrder = {'item': 'METALS', 'src': 'OE-PM-TR', 'dest': 'OE-PM', 'buy': 15, 'sell': 30, 'fuel': 1, 'gain': 15}
initialOE = {'dest': 'OE-UC'}
initialZY1 = {'dest': 'ZY1-GH'}
ORDERS = [
    #(credit>=, shipCount<=, model, role)
    {'cred': 100 * 1000, 'shipC':0, 'model': 'EM-MK-I', 'role': 'scout', 'location': 'OE-PM-TR', 'initial': initialOrder},
    {'cred': 100 * 1000, 'shipC':2, 'model': 'GR-MK-I', 'role': 'OE', 'location': 'OE-PM-TR', 'initial': initialOrder},
    {'cred': 300 * 1000, 'shipC':3, 'model': 'GR-MK-III'},
    {'cred': 480 * 1000, 'shipC':4, 'loan': True},
    {'cred': 800 * 1000, 'shipC':6, 'model': 'HM-MK-III'},
    {'cred': 1200 * 1000, 'shipC':16, 'model': 'HM-MK-III'},
    {'cred': 1200 * 1000, 'shipC':16, 'roleSwitch': ('OE', 'scout')},
    {'cred': 3000 * 1000, 'shipC':24, 'model': 'HM-MK-III'},
    {'cred': 6000 * 1000, 'shipC':32, 'model': 'HM-MK-III'},
    # {'cred': 10 * 1000000, 'buildC': 0, 'minShipC':15, 'build': 'CHEMICAL_PLANT', 'location': 'OE-UC-OB'},
    # {'cred': 10 * 1000000, 'buildC': 0, 'minShipC':15, 'build': 'SHIPYARD', 'location': 'OE-UC'},
    # {'cred': 10 * 1000000, 'buildC': 1, 'minShipC':20, 'build': 'MINE', 'location': 'OE-UC-AD'},
    # {'cred': 10 * 1000000, 'shipC':36, 'model': 'GR-MK-III', 'role': 'structure', 'initial': initialOE},
    {'cred': 200 * 1000000, 'buildC': 0, 'shipC':40, 'model': 'HM-MK-III'},
    # {'cred': 2 * 1000000, 'buildC': 3, 'shipC':17, 'model': 'DR-MK-I'}, # Metals - Rare
    # {'cred': 2 * 1000000, 'buildC': 3, 'shipC':18, 'model': 'TD-MK-I'}, # Fuel
    {'cred': 200 * 1000000, 'buildC': 0, 'build': 'RESEARCH_OUTPOST', 'location': 'ZY1-GH'},
    {'cred': 200 * 1000000, 'buildC': 1, 'build': 'ELECTRONICS_FACTORY', 'location': 'ZY1-GH-HD'},
    {'cred': 200 * 1000000, 'buildC': 2, 'build': 'FABRICATION_PLANT', 'location': 'ZY1-GH-NT'},
    {'cred': 200 * 1000000, 'buildC': 3, 'build': 'CHEMICAL_PLANT', 'location': 'ZY1-GH-MD'},
    {'cred': 200 * 1000000, 'buildC': 4, 'build': 'DRONE_FACTORY', 'location': 'ZY1-GG'},
    {'cred': 200 * 1000000, 'buildC': 5, 'build': 'MINE', 'location': 'ZY1-GG-CY'},
    {'cred': 200 * 1000000, 'buildC': 6, 'build': 'RARE_EARTH_MINE', 'location': 'ZY1-T85'},
    {'cred': 200 * 1000000, 'buildC': 7, 'build': 'EXPLOSIVES_FACILITY', 'location': 'ZY1-GG-LAO'},
    {'cred': 200 * 1000000, 'shipC':48, 'model': 'GR-MK-III', 'role': 'structure', 'initial': initialZY1},
]

class State:
    def __init__(self, username=USERNAME, token=TOKEN, galaxies={'OE': {}}, distances={}, locations={}, loans={},
                 goods={}, ships={}, stats={}, structures={}, market={}, orders={'_live': {}, '_rank': {}}, my={}):
        self.username = username
        self.token = token
        self.galaxies = galaxies
        self.locations = locations
        self.loans = loans
        self.goods = goods
        self.ships = ships
        self.stats = stats
        self.structures = structures
        self.distances = distances # departure|destination: (fuel, time)
        self.market = market
        self.orders = orders
        self.my = my
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=2)

st = State()

############ SAVE LOAD ############
def loadJson():
    global st
    try:
        with open(SAVE, 'r') as f:
            loadJson = json.load(f)
            st = State(**loadJson)
    except Exception:
        print('no save data, continuing')

def saveJson(fileName):
    with open(fileName, 'w') as f:
        f.write(st.toJSON())

def callApi(method, url, data={}, noToken=False):
    if not noToken:
        data['token'] = st.token
    if method == 'GET':
        js = requests.get(HOST + url, params=data).json()
    if method == 'POST':
        js = requests.post(HOST + url, params=data).json()
    if method == 'PUT':
        js = requests.put(HOST + url, params=data).json()
    if method == 'DELETE':
        js = requests.delete(HOST + url, params=data).json()
    time.sleep(REQUEST_SLEEP)
    if 'error' in js:
        print('FAIL', method, url, data, js)
    return js

############ RESET LOAN ############
def reset():
    global st, TOKEN
    st = State()
    while len(st.loans) == 0:
        js = callApi('POST', '/users/%s/claim' % st.username, noToken=True)
        if 'token' in js:
            print('user creation:', js)
            TOKEN = js['token']
            st.token = TOKEN
            availableLoans()
            getLoan(list(st.loans.keys())[0])
            updateLoans()
        elif TOKEN == '':
            print(js)
            time.sleep(60)
            continue
        availableLoans()
        availableGoods()
        availableShips()
        availableStructures()
        for glx in st.galaxies:
            updateGalaxy(glx)

def getLoan(loanType):
    js = callApi('POST', '/my/loans', {'type': loanType})
    print('get loan:', js)

def payLoan(loanId):
    js = callApi('PUT', '/my/loans/' + loanId)
    print('loan paid:', js)

def updateLoans():
    js = callApi('GET', '/my/loans/')
    if 'loans' in js:
        st.loans['active'] = [i for i in js['loans'] if i['status'] != 'PAID']
        st.loans['old'] = [i for i in js['loans'] if i['status'] == 'PAID']

def availableLoans():
    js = callApi('GET', '/types/loans')
    if 'loans' in js:
        st.loans = {i['type']:i for i in js['loans']}

def availableGoods():
    js = callApi('GET', '/types/goods')
    if 'goods' in js:
        st.goods = {i['symbol']:i for i in js['goods']}

def availableShips():
    js = callApi('GET', '/types/ships')
    if 'ships' in js:
        st.ships = {i['type']:i for i in js['ships']}

def availableStructures():
    js = callApi('GET', '/types/structures')
    if 'structures' in js:
        st.structures = {i['type']:i for i in js['structures']}

def updateGalaxy(galaxy='OE'):
    js = callApi('GET', '/systems/%s/ship-listings' % galaxy)
    st.galaxies[galaxy] = {}
    if 'shipListings' in js:
        for ship in js['shipListings']:
            if 'purchaseLocations' not in st.ships[ship['type']]:
                st.ships[ship['type']]['purchaseLocations'] = {}
            st.ships[ship['type']]['purchaseLocations'].update({i['location']:i['price'] for i in ship['purchaseLocations']})
    js = callApi('GET', '/systems/%s/locations' % galaxy)
    if 'locations' in js:
        st.locations.update({i['symbol']:i for i in js['locations']})
        for warp in js['locations']:
            parts = warp['symbol'].split('-W-')
            if len(parts) > 1:
                currentG = parts[0]
                newW = parts[1]
                st.galaxies[currentG][newW] = [warp['symbol']]
                for g in st.galaxies.get(newW, {}).keys():
                    if g != currentG:
                        st.galaxies[currentG][g] = st.galaxies[currentG][newW].copy()
                        st.galaxies[currentG][g].extend(st.galaxies[newW][g])
                for g in [i for i in st.galaxies.keys() if i != currentG and i != newW]:
                    if newW not in st.galaxies[g]:
                        st.galaxies[g][newW] = st.galaxies[g][currentG].copy()
                        st.galaxies[g][newW].append(warp['symbol'])

############ LowLvl Requests ############
def buyShip(sType, location):
    js = callApi('POST', '/my/ships', {'location': location, 'type': sType})
    print('ship bought:', js)
    return js

def sellShip(shipId):
    js = callApi('DELETE', '/my/ships/%s/' % shipId)
    print('ship bought:', js)

def buyItem(ship, good, quantity, location):
    shipId = ship['id']
    loadSpeed = ship['loadingSpeed']
    print('buying %s %d for %s' % (good, quantity, shipId))
    while (quantity > loadSpeed):
        js = callApi('POST', '/my/purchase-orders', {'shipId': shipId, 'good': good, 'quantity': loadSpeed})
        quantity -= loadSpeed
    js = callApi('POST', '/my/purchase-orders', {'shipId': shipId, 'good': good, 'quantity': quantity})
    st.stats['buy_' + good + '_' + location] = st.stats.get('buy_' + good + '_' + location, 0) + quantity
    if 'user' in js and 'credits' in js['user']:
        st.my['credits'] = js['credits']
    return js

def sellItem(ship, good, quantity, location):
    shipId = ship['id']
    loadSpeed = ship['loadingSpeed']
    print('selling %s %d for %s' % (good, quantity, shipId))
    while (quantity > loadSpeed):
        js = callApi('POST', '/my/sell-orders', {'shipId': shipId, 'good': good, 'quantity': loadSpeed})
        quantity -= loadSpeed
    js = callApi('POST', '/my/sell-orders', {'shipId': shipId, 'good': good, 'quantity': quantity})

    st.stats['sell_' + good + '_' + location] = st.stats.get('sell_' + good + '_' + location, 0) + quantity
    if js.get('order', {}).get('total') and shipId in st.orders['_live']:
        print('trade summary %s %d for %d' % (good, quantity, js['order']['total'] - st.orders['_live'][shipId]['price']))
        del st.orders['_live'][shipId]
    if 'user' in js and 'credits' in js['user']:
        st.my['credits'] = js['credits']

def transferItem(fromShipId, toShipId, good, quantity):
    print('transfer %s -> %s %s %d' % (fromShipId, toShipId, good, quantity))
    return callApi('POST', '/my/ships/%s/transfer' % fromShipId, {'shipId': toShipId, 'good': good, 'quantity': quantity})

def deleteItem(shipId, good, quantity):
    print('jettison %s -> %s %d' % (shipId, good, quantity))
    return callApi('POST', '/my/ships/%s/jettison' % shipId, {'good': good, 'quantity': quantity})

def buildStructure(structureType, location):
    print('buildStructure %s to %s' % (structureType, location))
    return callApi('POST', '/my/structures', {'location': location, 'type': structureType})

def shipToStructure(ship, structureId, good, quantity, location):
    shipId = ship['id']
    loadSpeed = ship['loadingSpeed']
    print('shipToStructure from %s to %s %s %d' % (shipId, structureId, good, quantity))
    while (quantity > loadSpeed):
        js = callApi('POST', '/my/structures/%s/deposit' % structureId, {'shipId': shipId, 'good': good, 'quantity': loadSpeed})
        quantity -= loadSpeed
    js = callApi('POST', '/my/structures/%s/deposit' % structureId, {'shipId': shipId, 'good': good, 'quantity': quantity})

    st.stats['deposit_' + good + '_' + location] = st.stats.get('deposit_' + good + '_' + location, 0) + quantity
    if 'error' not in js and shipId in st.orders['_live']:
        del st.orders['_live'][shipId]
    return js

def structureToShip(ship, structureId, good, quantity, location):
    shipId = ship['id']
    loadSpeed = ship['loadingSpeed']
    print('structureToShip from %s to %s %s %d' % (shipId, structureId, good, quantity))
    while (quantity > loadSpeed):
        js = callApi('POST', '/my/structures/%s/transfer' % structureId, {'shipId': shipId, 'good': good, 'quantity': loadSpeed})
        quantity -= loadSpeed
    js = callApi('POST', '/my/structures/%s/transfer' % structureId, {'shipId': shipId, 'good': good, 'quantity': quantity})

    st.stats['transfer_' + good + '_' + location] = st.stats.get('transfer_' + good + '_' + location, 0) + quantity
    return js

def setPath(shipId, destination, mkI):
    print('sending %s to %s' % (shipId, destination))
    js = callApi('POST', '/my/flight-plans', {'shipId': shipId, 'destination': destination})
    if 'error' not in js:
        st.distances[js['flightPlan']['departure'] + '|' + js['flightPlan']['destination'] + '|' + str(mkI)] = (js['flightPlan']['fuelConsumed'], js['flightPlan']['timeRemainingInSeconds'])

def warpPath(shipId, mkI):
    print('warping %s' % (shipId))
    js = callApi('POST', '/my/warp-jumps', {'shipId': shipId})
    if 'error' not in js:
        st.distances[js['flightPlan']['departure'] + '|' + js['flightPlan']['destination'] + '|' + str(mkI)] = (js['flightPlan']['fuelConsumed'], js['flightPlan']['timeRemainingInSeconds'])

############ Helper Utils ############
def isSameSystem(source, dest):
    return source.split('-', 1)[0] == dest.split('-', 1)[0]

def getMaxFuelNeed(orders, shipType='MK-II'):
    if orders:
        return max(map(lambda x: fuelConsumptionIntra(x['src'], x['dest'], shipType), orders))
    return 0

def getTotalFuelConsumption(source, destination, shipType='MK-II'):
    total = 0
    for path in createPathOrders(source, destination):
        if path.get('type') != 'warp':
            total += fuelConsumptionIntra(path['src'], path['dest'], shipType)
    return total

def fuelConsumptionIntra(source, destination, shipType='MK-II'):
    if destination == 'warp':
        return 0
    xDiff = st.locations[source]['x'] - st.locations[destination]['x']
    yDiff = st.locations[source]['y'] - st.locations[destination]['y']
    return math.ceil(math.sqrt(xDiff * xDiff + yDiff * yDiff) / 7.5) + len(shipType.rsplit('-', 1)[-1]) + (2 if st.locations[source]['type'] == 'PLANET' else 0)

def getWormholePath(planetSrc, planetDest=None):
    srcG = planetSrc.split('-', 1)[0]
    if planetDest is None:
        for destG in st.galaxies[srcG].keys():
            if destG not in st.galaxies:
                return st.galaxies[srcG][destG]
    else:
        destG = planetDest.split('-', 1)[0]
        return st.galaxies[srcG][destG]
    return [random.choice([i for i in st.locations.keys() if i.startswith(planetSrc.split('-')[0] + '-W-')])]

def fuelPrice(source):
    if source in st.goods['FUEL']:
        return st.goods['FUEL'][source]['purchasePricePerUnit']
    return (st.goods['FUEL']['min'] + st.goods['FUEL']['max']) / 2

def tradeWeight(trade, ship):
    if ship['maxCargo'] < trade['fuel'] * 2:
        return 0
    netGain = (ship['maxCargo'] - trade['fuel']) * trade['gain'] - trade['fuel'] * fuelPrice(trade['src'])
    netGain -= getTotalFuelConsumption(ship['location'], trade['src'], ship['type']) * fuelPrice(ship['location'])
    if netGain < 0:
        return 0
    # return (trade['gain']+5)**2 if trade['src'] == ship['location'] else trade['gain']**2
    return (trade['gain']+5) if trade['src'] == ship['location'] else trade['gain']

def createPathOrders(src, dest):
    path = []
    if src == dest:
        return path
    if dest == 'warp':
        wormHoleId = getWormholePath(src)[0]
        if src != wormHoleId:
            path.append({'src': src, 'dest': wormHoleId})
        path.append({'src': wormHoleId, 'dest': 'warp', 'type': 'warp'})
    elif isSameSystem(src, dest):
        path.append({'src': src, 'dest': dest})
    else:
        wormHolePath = getWormholePath(src, dest)
        current = src
        for lpath in zip([src] + wormHolePath, wormHolePath):
            if current != lpath[1]:
                path.append({'src': current, 'dest': lpath[1]})
            path.append({'src': lpath[1], 'dest': lpath[1], 'type': 'warp'})
            current = '-'.join(reversed(lpath[1].split('-')))
        path.append({'src': current, 'dest': dest})
    return path

def printLeaderboard():
    js = callApi('GET', '/game/leaderboard/net-worth')
    for node in js.get('netWorth', []):
        print('%20s\t%10s\t%2d' % (node['username'], node['netWorth'], node['rank']))
    current = int(time.time())
    for market, mJs in st.market.items():
        print(market, current - mJs.get('_update'))

def printGoods():
    for good, gJs in st.goods.items():
        print(good)
        for loca, dJs in gJs.items():
            if loca in st.locations:
                print('  %9s\t%5d %2d %6d' % (loca, dJs['pricePerUnit'], dJs['spread'], dJs['quantityAvailable']))

def printShips():
    print('MODEL         SPEED CARGO LOADING')
    for ship, sJs in st.ships.items():
        print('%9s\t%3d %4d %4d\t%s %s' % (sJs['type'], sJs['speed'], sJs['maxCargo'], sJs['loadingSpeed'], ' '.join(['%s(%s)' % (i[0], i[1]) for i in sJs.get('purchaseLocations', {}).items()]), ' '.join(sJs.get('restrictedGoods', []))))

def printAvailableStructures():
    for sJs in st.structures.values():
        print('%s (%10d) %s -> %s %s' % (sJs['type'], sJs['price'], sJs['consumes'], sJs['produces'], sJs['allowedPlanetTraits']))
        locas = []
        for lJs in st.locations.values():
            traitFound = True
            for trait in sJs.get('allowedPlanetTraits', []):
                traitFound = False
                for ptrait in lJs.get('traits', []):
                    # prefixes: SOME_ ABUNDANT_ '' 
                    if trait in ptrait:
                        traitFound = True
                if traitFound is False:
                    break
            if traitFound is False:
                continue
            if lJs['type'] in sJs['allowedLocationTypes']:
                locas.append('%s(%s)' % (lJs['symbol'], lJs['type'][:2]))
        print('  ' + ' '.join(locas))

############ Main ############
def updateStatus():
    js = callApi('GET', '/my/account')
    if 'user' in js:
        js = js['user']
        js['_item_sum'] = 0
        for onBoard in st.orders['_live'].values():
            js['_item_sum'] += onBoard.get('price', 0)
        print('=== %s %d [%d](%d) %d -- %d' % (datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S'),
                js['credits'], js['_item_sum'], js['shipCount'], js['structureCount'], js['credits'] + js['_item_sum']))
    if js.get('shipCount', 0) > 0:
        sjs = callApi('GET', '/my/ships')
        if 'ships' in sjs:
            js['_active_locations'] = set()
            js['ships'] = sjs['ships']
            for ship in sjs['ships']:
                if 'location' in ship:
                    js['_active_locations'].add(ship['location'])
                fuel = 0
                cargoStr = ''
                for cargo in ship['cargo']:
                    if cargo['good'] == 'FUEL':
                        fuel = cargo['quantity']
                        continue
                    cargoStr += '%s:%d(%d)' % (cargo['good'], cargo['quantity'], cargo['totalVolume'])
                currentTrade = ""
                if ship['id'] in st.orders['_live']:
                    currentTrade = tradeStr(st.orders['_live'][ship['id']])
                print('(%8s) %9s[%3s]%29s\t%s f:%d %s' % (ship.get('location', ''), ship['type'], st.orders['_rank'].get(ship['id'], '')[:3], ship['id'], currentTrade, fuel, cargoStr))
            js['_active_locations'] = list(js['_active_locations'])
            for location in js['_active_locations']:
                if location.split('-')[0] not in st.galaxies:
                    updateGalaxy(location.split('-')[0])
                getMarketplace(location)
    if js.get('structureCount', 0) > 0:
        sjs = callApi('GET', '/my/structures')
        if 'structures' in sjs:
            js['structures'] = sjs['structures']
            for structure in sjs['structures']:
                cargoStr = ''
                for inv in structure['inventory']:
                    cargoStr += ' %s:%d' % (inv['good'], inv['quantity'])
                print('(%8s) %9s %s\t%s' % (structure['location'], structure['type'], structure['id'], cargoStr))
    return js

def getMarketplace(location):
    js = callApi('GET', '/locations/%s/marketplace' % location)
    if 'marketplace' in js:
        mjs = {i['symbol']:i for i in js['marketplace']}
        mjs['_update'] = int(time.time())
        st.market[location] = mjs
        for item in js['marketplace']:
            if item['symbol'] not in st.goods:
                print ('good type unknown', item)
            st.goods[item['symbol']][location] = item
            st.goods[item['symbol']]['min'] = min(st.goods[item['symbol']].get('min', item['pricePerUnit']), item['pricePerUnit'])
            st.goods[item['symbol']]['max'] = max(st.goods[item['symbol']].get('max', item['pricePerUnit']), item['pricePerUnit'])
            st.goods[item['symbol']]['minSpread'] = min(st.goods[item['symbol']].get('minSpread', item['spread']), item['spread'])
            st.goods[item['symbol']]['maxSpread'] = max(st.goods[item['symbol']].get('maxSpread', item['spread']), item['spread'])

def sellAndDeposit():
    for ship in st.my.get('ships', []):
        if 'location' in ship:
            if len(st.orders.get(ship['id'], [])) == 0:
                for cargo in ship['cargo']:
                    if cargo['good'] != 'FUEL':
                        if ship['id'] in st.orders['_live']:
                            lastOrder = st.orders['_live'][ship['id']]
                            if 'toStructureId' in lastOrder:
                                shipToStructure(ship, lastOrder['toStructureId'], cargo['good'], cargo['quantity'], ship['location'])
                                ship['spaceAvailable'] += cargo['totalVolume']
                            else:
                                sellItem(ship, cargo['good'], cargo['quantity'], ship['location'])
                                ship['spaceAvailable'] += cargo['totalVolume']
                        else:
                            sellItem(ship, cargo['good'], cargo['quantity'], ship['location'])
                            ship['spaceAvailable'] += cargo['totalVolume']


def decideExpansion():
    shipCount = len(st.my.get('ships', []))
    buildCount = len(st.my.get('structures', []))
    planetCount = len(st.market.keys())
    for order in ORDERS:
        if order.get('minShipC', 0) <= shipCount <= order.get('shipC', 48) and \
           buildCount == order.get('buildC', buildCount) and st.my['credits'] > order.get('cred', 0):
            if 'model' in order:
                # buy ship
                if 'location' in order:
                    js = buyShip(order['model'], order['location'])
                    if 'ship' in js:
                        st.my['credits'] = js['credits']
                        st.orders['_rank'][js['ship']['id']] = order.get('role', 'inter')
                        if 'initial' in order:
                            st.orders[js['ship']['id']] = [order['initial']]
                else:
                    for store in set(st.ships[order['model']]['purchaseLocations'].keys()).intersection(st.my.get('_active_locations', [])):
                        js = buyShip(order['model'], store)
                        if 'ship' in js:
                            st.my['credits'] = js['credits']
                            st.orders['_rank'][js['ship']['id']] = order.get('role', 'inter')
                            if 'initial' in order:
                                st.orders[js['ship']['id']] = [order['initial']]
            elif 'build' in order:
                # build structure
                if order['location'] in st.my.get('_active_locations', []):
                    js = buildStructure(order['build'], order['location'])
                    if 'error' not in js and 'credits' in js:
                        st.my['credits'] = js['credits']
            elif 'loan' in order and st.loans.get('active') and st.loans['active'][0].get('id'):
                payLoan(st.loans['active'][0]['id'])
                updateLoans()
            elif 'roleSwitch' in order:
                for ship in st.my.get('ships', []):
                    if st.orders['_rank'][ship['id']] == order['roleSwitch'][0]:
                        st.orders['_rank'][ship['id']] = order['roleSwitch'][1]


def findTrades():
    trades = []
    # TODO structure to structure trade -- 'OE-UC-AD' -> 'OE-UC' metal
    for structure in st.my.get('structures', []):
        inventory = {v['good']: v['quantity'] for v in structure['inventory']}
        for good in structure['consumes']:
            if inventory[good] < 50: # TODO 100K produce varsa alma ?
                gJs = st.goods[good]
                marketIds = set(gJs.keys()).intersection(st.locations.keys())
                toLocId = structure['location']
                for fromLocId in marketIds:
                    trades.append({
                        'type': 'intra' if isSameSystem(fromLocId, toLocId) else 'inter',
                        'toStructureId': structure['id'],
                        'item': good,
                        'src': fromLocId,
                        'dest': toLocId,
                        'quantityAvailable': gJs[fromLocId]['quantityAvailable'],
                        'buy': gJs[fromLocId]['purchasePricePerUnit'],
                        'sell': gJs[fromLocId]['purchasePricePerUnit'] * 2,
                        'fuel': getTotalFuelConsumption(fromLocId, toLocId),
                        'gain': (1.0 * gJs[fromLocId]['purchasePricePerUnit'] / gJs['volumePerUnit'])})
        for good in structure['produces']:
            if inventory[good] > 1000:
                gJs = st.goods[good]
                marketIds = set(gJs.keys()).intersection(st.locations.keys())
                fromLocId = structure['location']
                for toLocId in marketIds:
                    trades.append({
                        'type': 'intra' if isSameSystem(fromLocId, toLocId) else 'inter',
                        'fromStructureId': structure['id'],
                        'item': good,
                        'src': fromLocId,
                        'dest': toLocId,
                        'quantityAvailable': inventory[good],
                        'buy': 0,
                        'sell': gJs[toLocId]['sellPricePerUnit'],
                        'fuel': getTotalFuelConsumption(fromLocId, toLocId),
                        'gain': (1.0 * gJs[toLocId]['sellPricePerUnit'] / gJs['volumePerUnit'])})

    for item, gJs in st.goods.items():
        if item == 'FUEL':
            continue
        marketIds = set(gJs.keys()).intersection(st.locations.keys())
        for fromLocId in marketIds:
            for toLocId in marketIds:
                if gJs[fromLocId]['purchasePricePerUnit'] < gJs[toLocId]['sellPricePerUnit']:
                    trades.append({
                        'type': 'intra' if isSameSystem(fromLocId, toLocId) else 'inter',
                        'item': item,
                        'src': fromLocId,
                        'dest': toLocId,
                        'quantityAvailable': gJs[fromLocId]['quantityAvailable'],
                        'buy': gJs[fromLocId]['purchasePricePerUnit'],
                        'sell': gJs[toLocId]['sellPricePerUnit'],
                        'fuel': getTotalFuelConsumption(fromLocId, toLocId),
                        'gain': (1.0 * (gJs[toLocId]['sellPricePerUnit'] - gJs[fromLocId]['purchasePricePerUnit']) / gJs['volumePerUnit'])})
    return sorted(trades, key=lambda x: x['gain'], reverse=True) # [(gain/volume, s, d, item), ... ]

def tradeStr(trade):
    return trade.get('src', '') + " > " + trade.get('dest', '') + " : " + trade.get('item', '')

def decideTrades(trades):
    currentTime = int(time.time())
    currentTrades = Counter(map(tradeStr, st.orders['_live'].values()))
    for ship in st.my.get('ships', []):
        if 'location' in ship and st.orders.get(ship['id'], []) == []:
            if st.orders['_rank'][ship['id']] == 'scout':
                unexplored = list(st.locations.keys() - st.market.keys())
                expiredMarket = [m for m, mJs in st.market.items() if currentTime - mJs.get('_update') > 1200 and '-W-' not in m]
                dest = ''
                if unexplored:
                    dest = random.choices(unexplored, list(map  (lambda d: 100 - getTotalFuelConsumption(ship['location'], d), unexplored)))
                    if dest:
                        print('scouting %s' % dest[0])
                else:
                    if len(st.loans.get('active', [''])) == 0:
                        unexploredGalaxy = [g for g in st.galaxies['OE'].keys() if g not in st.galaxies]
                        if unexploredGalaxy:
                            dest = ['warp']
                if expiredMarket and dest == '':
                    try:
                        dest = random.choices(expiredMarket, list(map  (lambda d: 100 - getTotalFuelConsumption(ship['location'], d), expiredMarket)))
                        if dest:
                            print('expiredMarket %s' % dest[0])
                    except Exception:
                        print ('expired market random patladi', expiredMarket)
                if dest:
                    st.orders[ship['id']] = createPathOrders(ship['location'], dest[0])
                    continue

            if st.orders['_rank'][ship['id']] in st.galaxies:
                # galaxy bound ship
                filteredTrades = list(filter(lambda x: x['type'] == 'intra' and isSameSystem(x['src'], st.orders['_rank'][ship['id']]) and currentTrades[tradeStr(x)] < 2, trades))
            elif st.orders['_rank'][ship['id']].startswith('structure'):
                filteredTrades = list(filter(lambda x: x['type'] == 'intra' and \
                                    isSameSystem(x['src'], ship['location']) and \
                                    ('fromStructureId' in x or 'toStructureId' in x), trades))
                if not filteredTrades:
                    filteredTrades = list(filter(lambda x: x['type'] == 'intra' and isSameSystem(x['src'], ship['location']), trades))
                if not filteredTrades:
                    filteredTrades = list(filter(lambda x: currentTrades[tradeStr(x)] < (2 if x['type'] == 'intra' else 1), trades))
            else:
                filteredTrades = list(filter(lambda x: currentTrades[tradeStr(x)] < (2 if x['type'] == 'intra' else 1), trades))

            # TODO structure tradelerde quantity dikkat et
            try:
                trade = random.choices(filteredTrades, list(map(lambda x: tradeWeight(x, ship), filteredTrades)))
                if trade:
                    st.orders[ship['id']] = createPathOrders(ship['location'], trade[0]['src'])
                    st.orders[ship['id']].append(trade[0])
                    st.orders['_live'][ship['id']] = dict({'quantity': 0, 'price': 0}, **trade[0])
                    currentTrades[tradeStr(trade[0])] += 1
            except Exception:
                print('random failed', filteredTrades, list(map(lambda x: tradeWeight(x, ship), filteredTrades)))

def executeOrders():
    # st.orders = # shipId: [('src', 'dest', 'items', 'quantity'), ...]
    for ship in st.my.get('ships', []):
        if 'location' in ship and st.orders.get(ship['id'], []) != []:
            order = st.orders[ship['id']].pop(0)
            if isSameSystem(ship['location'], order['dest']):
                if 'src' not in order:
                    order['src'] = ship['location']
                st.orders[ship['id']].insert(0, order) # pathOrder = order
            else:
                st.orders[ship['id']] = createPathOrders(ship['location'], order['dest'])

            activeFuel = 0
            for cargo in ship['cargo']:
                if cargo['good'] == 'FUEL':
                    activeFuel = cargo['quantity']
            maxFuelNeeded = getMaxFuelNeed(st.orders[ship['id']], ship['type'])
            if maxFuelNeeded - activeFuel > 0:
                buyItem(ship, 'FUEL', maxFuelNeeded - activeFuel, ship['location'])
            if order.get('item'):
                quantity = min((int(ship['maxCargo'] - max(maxFuelNeeded, activeFuel)) / st.goods[order['item']]['volumePerUnit']), order.get('quantityAvailable', 5000))
                if 'fromStructureId' in order:
                    retJs = structureToShip(ship, order['fromStructureId'], order['item'], quantity, ship['location'])
                else:
                    quantity = min(quantity, int(st.my['credits'] / (order['buy'] + 1)))
                    retJs = buyItem(ship, order['item'], quantity, ship['location'])
                if 'error' in retJs:
                    st.orders[ship['id']] = []
                    continue
                st.orders['_live'][ship['id']] = dict({'quantity': quantity, 'price': quantity * retJs.get('order', {}).get('pricePerUnit', 0)}, **order)

            if len(st.orders[ship['id']]) == 1 and order['dest'] == ship['location']: # zaten destinationdayiz
                if ship['id'] in st.orders['_live'] and 'item' in order:
                    # structureToShip sonrasi ayni yerde satacam;
                    sellItem(ship, order['item'], st.orders['_live'][ship['id']]['item'], ship['location'])
                continue

            pathOrder = st.orders[ship['id']].pop(0)
            if pathOrder.get('type', '') == 'warp':
                warpPath(ship['id'], len(ship['type'].rsplit('-', 1)[-1]))
            else:
                setPath(ship['id'], pathOrder['dest'], len(ship['type'].rsplit('-', 1)[-1]))

def main():
    global USERNAME, TOKEN, st
    parser = argparse.ArgumentParser(description='Twitter streaming over Twitter Search API')
    parser.add_argument('--reset', help='call reset at start', action='store_true')
    parser.add_argument('-i', '--info', help='print info and exit', action='store_true')
    parser.add_argument('-l', '--leader', help='print leaders and exit', action='store_true')
    parser.add_argument('-s', '--sleep', help='sleep seconds', default=20, type=int)
    parser.add_argument('--username', help='set username')
    parser.add_argument('--token', help='set token')
    args = parser.parse_args()
    if args.username:
        USERNAME = args.username
    if args.token:
        TOKEN = args.token
    st = State()

    loadJson()
    if args.reset or len(st.loans) == 0:
        reset()
    if args.info:
        printAvailableStructures()
        printGoods()
        printShips()
        return
    if args.leader:
        printLeaderboard()
        return

    loopCtr = 1
    while loopCtr > 0:
        try:
            status = updateStatus()
            if 'error' not in status:
                st.my = status
                if loopCtr % 60 == 0 and len(st.loans.get('active', [''])) > 0: # 20 * 30 = 1200sec -> 20dk+
                    updateLoans()
                sellAndDeposit()
                decideExpansion()
                trades = findTrades()
                decideTrades(trades)
                executeOrders()
                saveJson(SAVE)
            else:
                print('status failed')
                if status['error']['code'] == 40101:
                    # invalid token; create user reset all;
                    saveJson(datetime.datetime.now().strftime('%y%m%d') + '.json')
                    TOKEN = ''
                    reset()
        except Exception as e:
            print('main loop failed', e)
            traceback.print_exc()

        loopCtr += 1
        print('sleeping', args.sleep)
        time.sleep(args.sleep)

# https://kurt1288.github.io/Vocivos/ -- GUI
# https://deliverance.forcookies.dev
# https://spacetraders.staffordwilliams.com

if __name__ == '__main__':
    main()
