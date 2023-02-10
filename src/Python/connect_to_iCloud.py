'''
connect_to_iCloud.py




source ~/.bash_profile && python3 -m pip install pyicloud --upgrade


To setup, in the terminal:
icloud --username=carlos.aguilar.palacios@gmail.com
to delete it:
icloud --username=carlos.aguilar.palacios@gmail.com --delete-from-keyring

Follow:
https://github.com/picklepete/pyicloud

'''

#0.10.2

import utils.file_utils as fu
from pyicloud import PyiCloudService

api = PyiCloudService('carlos.aguilar.palacios@gmail.com')
for device in api.devices:
  print(device)
  fu.printJSON(device.status())
  fu.printJSON(device.content)


# access contacts
for c in api.contacts.all():
  print(c.get('firstName'), c.get('phones'))

api.drive.dir()

api.iphone

api.iphone.status()


# Access photos
#api.photos.all

#
api.files.dir()
pi.files['com~apple~Notes']





# api.trusted_devices()

#pyicloud.iphone.location()