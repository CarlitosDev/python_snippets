'''
    python_dynamic_programming.py

    Interestingly enough...the code in Medium is wrong.
    The one below reuses the dictionary.

'''


int_dict = {1:1}
def factorial_with_dp(n):
    if n in int_dict: 
        print(f'reading from {n} the cache')
        return int_dict[n]
    else:
        fact = n*factorial_with_dp(n-1)
        int_dict[n] = fact
    return fact


factorial_with_dp(1)
factorial_with_dp(4)

factorial_with_dp(12)
factorial_with_dp(13)