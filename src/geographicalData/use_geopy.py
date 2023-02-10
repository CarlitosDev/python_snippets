'''
use_geopy.py


 source ~/.bash_profile && pip3 install geopy

'''


user_agent="GIW"

from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent=user_agent)
location = geolocator.reverse("52.509669, 13.376294")


######

from geopy.geocoders import Nominatim
locator = Nominatim(user_agent = 'myGeocoder')
coordinates = '53.480837, -2.244914'
location = locator.reverse(coordinates)

address = location.raw['address']
print(address)

city = address.get('city', '')
state = address.get('state', '')
country = address.get('country', '')
code = address.get('country_code')
zipcode = address.get('postcode')
print('City : ',city)
print('State : ',state)
print('Country : ',country)
print('Zip Code : ', zipcode)
