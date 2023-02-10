import pyperclip as pp

book_titles = ['How to Talk to Anyone', 'The Power of Habit', 
'The 4-Hour Body', 'Rich Dad, Poor Dad', 'The High 5 Habit', 
'The Power of Now', 'The 10X Rule', 'Unlimited Memory', 'Super Human',
'Limitless', 'Think and Grow Rich', 'Quiet', 'Elon Musk', 
'The 80/20 principle', 'The 5 Love Languages', 'The 5 AM club', 
'Lean in', 'Never Eat Alone', 'Thinking fast and slow']


# for book_title in book_titles:
idx = -1

idx +=1
book_title = book_titles[idx]
pre_prompt = f'Could you write down the the author name and ten takeaways from the book {book_title}?'
print((pre_prompt))
pp.copy(pre_prompt)


pp.copy(f'{book_title} (book takeaways)')