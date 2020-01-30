>> Bunch: Access as JS objects/Matlab structs

from bunch import Bunch
# Adform settings
adform = Bunch()
adform.disable_ingest = False
adform.start_date = datetime(2018,06,27) # manual backfill before that date
adform.crawler = None  # 'mip_adform_live'


# My fix for Python3
pip3 install git+https://github.com/CarlitosDev/bunch.git