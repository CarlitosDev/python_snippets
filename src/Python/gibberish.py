gibberish.py


# show the keys of a dictionary
print('\n'.join(sorted(list(evc_api_info['meeting_components'].keys()))))

these_keys = evc_api_info['lesson_attendance']['attendances'][0]
print('\n'.join(sorted(list(these_keys))))


# assigments for dictionary keys.
import carlos_utils.string_utils as stru
for this_key in attender.keys():
  new_key = stru.snake_case_typer(this_key)
  print(f'''{new_key} = attender['{this_key}']''')

for this_key in userMeta.keys():
  new_key = stru.snake_case_typer(this_key)
  print(f'''{new_key} = userMeta['{this_key}']''')



# from code to dataclass

import pyperclip as pp
# use this one to create a dictionary
code_snippet = pp.paste()
for this_line in code_snippet.split('\n'):
  this_variable = this_line.split('=')[0].rstrip()
  print(f'''\'{this_variable}\':{this_variable},''')

d = {'center_code':center_code,
'user_id':user_id,
'video_un_mute':video_un_mute,
'video_display':video_display,
'video_toggle':video_toggle,
'attendance_token':attendance_token,
'role_code':role_code,
'enter_time':enter_time,
'exit_time':exit_time,
'user_device':user_device,
'external_user_id':external_user_id,
'attendance_ref_code':attendance_ref_code,
'display_name':display_name}


# for the dataclasses
for this_key, this_value in d.items():
  print(f'''{this_key}: {type(this_value).__name__}''')