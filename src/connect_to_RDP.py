'''
	connect_to_RDP.py


	https://github.com/diyan/pywinrm/

	pip3 install pywinrm requests_kerberos pykerberos
	pip3 install pywinrm[kerberos]

	

'''

#!/usr/bin/env python
import winrm


# Create winrm connection.
ip_remote_desktop = '10.162.67.30'
username = 'carlos.aguilar@ef.com'
password = '*Education*'


ip_remote_desktop = '10.162.97.131'
username = 'carlos.aguilar@ef.com'
password = ':endOfHoliday:'


# sess = winrm.Session('https://10.0.0.1', auth=('username', 'password'), transport='kerberos')
# result = sess.run_cmd('ipconfig', ['/all'])




sess = winrm.Session('https://' + ip_remote_desktop, auth=(username, password) )#, transport='kerberos')
result = sess.run_cmd('ipconfig', ['/all'])