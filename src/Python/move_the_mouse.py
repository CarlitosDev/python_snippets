'''
move_the_mouse.py

source ~/.bash_profile && python3 -m pip install pyautogui
source ~/.bash_profile && python3 move_the_mouse.py
source ~/.bash_profile && python3 '/Users/carlos.aguilar/Google Drive/PythonSnippets/Python/move_the_mouse.py'

# From here: https://github.com/daOyster/Mouse-Circle-Script/blob/master/mouseCircle.py


'''


import time
import math
import pyautogui

SIZE_X, SIZE_Y = pyautogui.size()


sleep_time = 100
max_time = 100*sleep_time
total_time = 0
STEPS = 1000
i = 0
while total_time < max_time:
	# Get the decimal coordinate of each 'tick' [0.0,1.0]
	# using sin/cos function
	j = (((i/STEPS)*2)*math.pi)
	x = math.cos(j) 
	y = math.sin(j) 
	# plot the mouse coordinates along a oval shape that
	# is centered on the middle of the screen.
	pyautogui.moveTo( SIZE_X/2 + (SIZE_Y/3)*x,SIZE_Y/2 + (SIZE_Y/3)*y, duration=0.1)
	time.sleep(sleep_time)
	total_time += sleep_time
	i+=1