'''

cool_print_with_colorama.py

foreground for the text color
background for the background color
style for some additional color styling

'''


from colorama import Fore, Back, Style

print(Fore.GREEN)
print("Task completed")

print('hey')

print(Back.RED)
print("Error occurred!")

print(Style.DIM)
print("Not that important")

print(Style.RESET_ALL)