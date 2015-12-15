"""
Defines the data package, which contains data files for the
games/appids below.

:author: Matt Mulholland (mulhodm@gmail.com)
:date: November, 2014
"""
APPID_DICT = dict(Dota_2='570',
                  Counter_Strike_Global_Offensive='730',
                  Team_Fortress_2='440',
                  Grand_Theft_Auto_V='271590',
                  Football_Manager_2015='295270',
                  Sid_Meiers_Civilization_5='8930',
                  Garrys_Mod='4000',
                  Arma_3='107410',
                  The_Elder_Scrolls_V='72850',
                  Warframe='230410',
                  Counter_Strike='10',
                  sample='sample_id')


def parse_appids(appids: list) -> list:
    """
    Parse the command-line argument passed in with the `--appids` flag,
    exiting if any of the resulting IDs do not map to games in
    APPID_DICT.

    :param appids: game IDs
    :type appids: str

    :returns: list of game IDs
    :rtype: list

    :raises ValueError: if unrecognized `appid` found in input
    """

    appids = appids.split(',')
    for appid in appids:
        if not appid in APPID_DICT.values():
            raise ValueError('{0} not found in APPID_DICT. Exiting.'.format(appid))
    return appids
