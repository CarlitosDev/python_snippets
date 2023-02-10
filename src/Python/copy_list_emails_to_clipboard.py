'''
copy_list_emails_to_clipboard.py
'''



import re

regex = r"<(.*?)>"
test_str = '''Carlos Aguilar <carlos.aguilar@ef.com>; Giu Membrino (Consultant) <ext.giu.membrino@ef.com>; Radek Busz (Consultant) <ext.radek.busz@ef.com>; Harry Deng <harry.deng@ef.com>; Isaac Li <isaac.li@ef.com>; Murugan Dhanapal <murugan.dhanapal@ef.com>; Oscar Remedios <oscar.remedios@ef.com>; Ralph Lynch <ralph.lynch@ef.com>; Ranjitha s Hakkapakki <ranjithas.hakkapakki@ef.com>; Tim Hesse <tim.hesse@ef.com>'''
matches = re.finditer(regex, test_str, re.MULTILINE)
waitingTime = 4
import pyperclip as pp
import time
for matchNum, match in enumerate(matches, start=1):
  this_match = match.group().replace('<', '').replace('>', '')
  print(this_match)
  pp.copy(this_match)
  time.sleep(waitingTime)