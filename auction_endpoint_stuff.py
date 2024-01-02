import requests


def get_auctions(page: int) -> list:
    resp = requests.request("GET", "https://api.hypixel.net/skyblock/auctions?page=" + str(page), headers={}, data={})
    return resp.json()['auctions']


def get_all_auctions() -> list:
    resp = requests.request("GET", "https://api.hypixel.net/skyblock/auctions?page=" + str(0), headers={}, data={})
    li = resp.json()['auctions']
    for i in range(1, resp.json()['totalPages']):
        li += get_auctions(i)
    return li


def get_all_bins() -> list:
    return list(filter(lambda x: x['bin'], get_all_auctions()))
