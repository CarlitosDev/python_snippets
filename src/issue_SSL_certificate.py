
'''
  This is an issue related to BigSur and OpenSSL
'''

python3 -c 'import urllib.request; urllib.request.urlopen("https://pypi.org")'


python3 -m pip install --upgrade certifi
open /Applications/Python\ 3.9/Install\ Certificates.command

'''
  What I did prior to that it was to install OpenSSL using the tarball file and ./Configure
  but I am not sure if that resolved it.
'''