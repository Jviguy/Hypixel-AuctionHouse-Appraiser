import base64
import gzip
import os
from io import BytesIO
from nbt import nbt
import pandas as pd
from nbt.nbt import TAG_String

from auction_endpoint_stuff import get_all_bins


def auction_to_dataframe(auction: dict) -> pd.DataFrame:
    """
    Converts an auction to a dataframe
    :param auction: auction dict
    :return: dataframe
    """
    item_bytes = auction['item_bytes']
    nbtf = item_bytes_to_nbt(item_bytes)
    tag = nbtf['i'][0]['tag']['ExtraAttributes']
    data = [{
        'auction_id': auction['uuid'],
        'item_id': get_item_name_from_nbt(tag),
        'price': auction['starting_bid'],
        'tier': auction['tier'],
        'category': auction['category'],
        'enchantments': get_enchantments_from_nbt(tag),
        'reforge': get_reforge_from_nbt(tag),
        'upgrade_level': get_upgrade_level_from_nbt(tag),
        'hot_potato_count': get_hotpotato_count_from_nbt(tag),
        'recomb': recomb_from_nbt(tag),
        'unlocked_gem_slots': get_unlocked_gem_slots_from_nbt(tag),
        'slotted_gems': get_slotted_gems_from_nbt(tag),
    }]
    return pd.DataFrame(data)


def item_bytes_to_nbt(b: str) -> nbt.NBTFile:
    return nbt.NBTFile(buffer=BytesIO(gzip.decompress(base64.b64decode(b))))


def get_enchantments_from_nbt(t: nbt.TAG_COMPOUND) -> list:
    if 'enchantments' not in t:
        return []
    ret = []
    for i in t['enchantments']:
        ret.append(i + " " + str(t['enchantments'][i]))
    return ret


def get_reforge_from_nbt(t: nbt.TAG_COMPOUND) -> str:
    if 'modifier' not in t:
        return ""
    return t['modifier'].value


def get_item_name_from_nbt(t: nbt.TAG_COMPOUND) -> str:
    if 'id' not in t:
        return ""
    return t['id'].value


def get_upgrade_level_from_nbt(t: nbt.TAG_COMPOUND) -> int:
    if 'upgrade_level' not in t:
        return -1
    return t['upgrade_level'].value


def get_hotpotato_count_from_nbt(t: nbt.TAG_COMPOUND) -> int:
    if 'hot_potato_count' not in t:
        return -1
    return t['hot_potato_count'].value


def recomb_from_nbt(t: nbt.TAG_COMPOUND) -> bool:
    if 'rarity_upgrades' not in t:
        return False
    return t['rarity_upgrades'].value == 1


def get_unlocked_gem_slots_from_nbt(t: nbt.TAG_COMPOUND) -> list:
    if 'gems' not in t:
        return []
    t = t['gems']
    if 'unlocked_slots' not in t:
        return []
    li = []
    for i in t['unlocked_slots']:
        li.append(i.value)
    return li


def get_slotted_gems_from_nbt(t: nbt.TAG_COMPOUND) -> list:
    if 'gems' not in t:
        return []
    t = t['gems']
    li = []
    for ta in t:
        if isinstance(t[ta], TAG_String):
            li.append(ta + " " + str(t[ta]))
        else:
            if ta == "unlocked_slots":
                continue
            li.append(ta + " " + str(t[ta]["quality"]))
    return li


x = get_all_bins()
# Check if the file exists
if os.path.isfile("auctions.csv"):
    all_data = pd.read_csv("auctions.csv")
else:
    all_data = pd.DataFrame()  # Create an empty DataFrame if the file doesn't exist

# Convert each item in 'x' to a DataFrame and store them in a list
list_of_df = [auction_to_dataframe(item) for item in x]

# Concatenate all DataFrames
all_data = pd.concat([all_data] + list_of_df, ignore_index=True)
all_data.to_csv("auctions.csv", index=False)
