'''
tricks_with_continue.py
'''

# Using continue passes to the next iteration of the for loop (avoiding an 'else')
d_distances = {'a':2, 'b':2}
for this_videofile in ['c', 'd', 'a', 'e', 'b']:
  if (this_videofile not in d_distances.keys()):
    continue
  print(f'Found {this_videofile}')