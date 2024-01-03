import base64
import datetime
import gzip
import json
import os
import sys
from io import BytesIO
from nbt import nbt
import pandas as pd
from nbt.nbt import TAG_String

from auction_endpoint_stuff import get_all_bins, get_auctions


def auction_to_dataframe(auction: dict) -> pd.DataFrame:
    """
    Converts an auction to a dataframe
    :param auction: auction dict
    :return: dataframe
    """
    item_bytes = auction['item_bytes']
    nbtf = item_bytes_to_nbt(item_bytes)
    tag = nbtf['i'][0]['tag']['ExtraAttributes']
    pet_info = get_pet_info_from_nbt(tag)
    rgb = get_rgb_from_nbt(tag)
    start_time = unix_to_skyblock_time(auction['start'])
    end_time = unix_to_skyblock_time(auction['end'])
    data = [{
        'auction_id': auction['uuid'],
        'item_count': get_item_count_from_nbt(nbtf['i'][0]),
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
        'pet_type': pet_info['type'],
        'pet_active': pet_info['active'],
        'pet_held_item': pet_info['heldItem'],
        'pet_exp': pet_info['exp'],
        'pet_candy_used': pet_info['candyUsed'],
        'dungeon_item_level': get_dungeon_item_level_from_nbt(tag),
        'red_armor_coloring': rgb[0],
        'green_armor_coloring': rgb[1],
        'blue_armor_coloring': rgb[2],
        'anvil_uses': get_anvil_uses_from_nbt(tag),
        'pelts_earned': get_pelts_earned_from_nbt(tag),
        'champion_combat_xp': get_champion_combat_xp_from_nbt(tag),
        'farmed_cultivating': get_farmed_cultivating_from_nbt(tag),
        'compact_blocks': get_compact_blocks_from_nbt(tag),
        'hecatomb_s_runs': get_hecatomb_s_runs_from_nbt(tag),
        'expertise_kills': get_expertise_kills_from_nbt(tag),
        'runes': get_runes_from_nbt(tag),
        'start_day': start_time['day'],
        'start_month': start_time['month'],
        'start_year': start_time['year'],
        'end_day': end_time['day'],
        'end_month': end_time['month'],
        'end_year': end_time['year']
    }]
    return pd.DataFrame(data)


def item_bytes_to_nbt(b: str) -> nbt.NBTFile:
    return nbt.NBTFile(buffer=BytesIO(gzip.decompress(base64.b64decode(b))))


def get_item_count_from_nbt(t: nbt.TAG_COMPOUND) -> int:
    if 'Count' not in t:
        return -1
    return t['Count'].value


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


def get_pet_info_from_nbt(t: nbt.TAG_COMPOUND) -> dict:
    if 'petInfo' not in t:
        return {"type": "", "exp": -1, "candyUsed": -1, "active": False, "heldItem": "", "tier": ""}
    t = json.loads(t['petInfo'].value)
    ret = {}
    if 'type' in t:
        ret['type'] = t['type']
    else:
        ret['type'] = ""
    if 'exp' in t:
        ret['exp'] = t['exp']
    else:
        ret['exp'] = -1
    if 'candyUsed' in t:
        ret['candyUsed'] = t['candyUsed']
    else:
        ret['candyUsed'] = -1
    if 'active' in t:
        ret['active'] = t['active']
    else:
        ret['active'] = False
    if 'heldItem' in t:
        ret['heldItem'] = t['heldItem']
    else:
        ret['heldItem'] = ""
    if 'tier' in t:
        ret['tier'] = t['tier']
    else:
        ret['tier'] = ""
    return ret


def get_dungeon_item_level_from_nbt(t: nbt.TAG_COMPOUND) -> int:
    if 'dungeon_item_level' not in t:
        return -1
    return t['dungeon_item_level'].value


def get_rgb_from_nbt(t: nbt.TAG_COMPOUND) -> tuple:
    if 'color' not in t:
        return -1, -1, -1
    t = t['color'].value
    t = list(map(int, t.split(":")))
    return t[0], t[1], t[2]


def get_anvil_uses_from_nbt(t: nbt.TAG_COMPOUND) -> int:
    if 'anvil_uses' not in t:
        return -1
    return t['anvil_uses'].value


def get_pelts_earned_from_nbt(t: nbt.TAG_COMPOUND) -> int:
    if 'pelts_earned' not in t:
        return -1
    return t['pelts_earned'].value


def get_champion_combat_xp_from_nbt(t: nbt.TAG_COMPOUND) -> int:
    if 'champion_combat_xp' not in t:
        return -1
    return t['champion_combat_xp'].value


def get_farmed_cultivating_from_nbt(t: nbt.TAG_COMPOUND) -> int:
    if 'farmed_cultivating' not in t:
        return -1
    return t['farmed_cultivating'].value


def get_compact_blocks_from_nbt(t: nbt.TAG_COMPOUND) -> int:
    if 'compact_blocks' not in t:
        return -1
    return t['compact_blocks'].value


def get_hecatomb_s_runs_from_nbt(t: nbt.TAG_COMPOUND) -> int:
    if 'hecatomb_s_runs' not in t:
        return -1
    return t['hecatomb_s_runs'].value


def get_expertise_kills_from_nbt(t: nbt.TAG_COMPOUND) -> int:
    if 'expertise_kills' not in t:
        return -1
    return t['expertise_kills'].value


def get_runes_from_nbt(t: nbt.TAG_COMPOUND) -> list:
    if 'runes' not in t:
        return []
    li = []
    for ta in t['runes']:
        li.append(ta)
    return li


def unix_to_skyblock_time(unix_timestamp):
    d = datetime.datetime.fromtimestamp(unix_timestamp / 1000.0)
    return {'day': d.day, 'month': d.month, 'year': d.year}


inp = input("Gather all data? (y/n)")
inp.strip()
inp = inp.lower()
if inp == "n":
    nb = item_bytes_to_nbt(
        "H4sIAAAAAAAAAFWS0Y6aQBSGR3e3VdO06ROUJpv0ygZ12e1eUhAZd2EqIgo3zQAjDAxCAHXxHfoKvfU9fLCm426bpleT/Oc75/9P5vQA6IIW7QEAWm3QpmHrRwtcKfl2U7d74KLG0QXo6jQkGsNRxalfPdCbp1vG0H5Dyg5owxBc3wZYDIZh2L8T/S/90ehW6t/7GPcDCUv3oXQ3JCLhfd/KvCBlTUnVBZ2aPNXbklTP1h1w5WC2Ja2fZJ9HUJmKeDlgwciK/ZVMoZpHhu0OzUN0MBJ3hOzx3mz2D1CRaaBPd17GKm/BUkjlW6jAkaHORE9lqTuEEppA0eV97lJLzAQezERjru1KbmZmaA4rhcoR3Hxt/KFX+BMHudz3Zc507Qw1MdiwA57U8ar5y5qFN5TiUHcaz5myYOUUQea8eOtWEy4XfziLEd0a8NrhpVad8553s+ciQ/9rZ95pfAVGiMoU65YYqPnucfRvxmM2KPzMSYJMy0JF2nqr2S6cODfPOeb31FDTkafKN+7SitFkmnp8RyOBN8ZhNnATLzVseWAmToLUheglYezZLDWSmeja4wapLDOW0wwt3cazIxHZs4FpR3sY5c/Z1jP+6uLDesb/qgdeh7QqGG664PIxL0mHi1fg+nS8m9clplFcC+syz4Q6JkJV8NMpP1VClm/r+DMnP5yOxDpD/YDRIBXqXNhRshdKEtCCVB858vZ0xKcjW5gKMgxkdsCliTMC3nNZLnEQbwgfqOFNxLO8Gz9xU7muS+pva1J1zjcM3siWrOjm+LsmmxMA2uCVijMcEXABwG+yoS4H8gIAAA\u003d\u003d")
    print(nb.pretty_tree())
else:
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
