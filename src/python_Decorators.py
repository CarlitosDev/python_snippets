'''

Cool way for measuring time in functions using a decorator

'''



'''
	(i) Define the decorator
'''

import functools
def timeFunction(function):
    """
        Use as a decorator function
        to time individual functions.
    """

    @functools.wraps(function)
    def wrapperTimer(*args, **kwargs):
        startTime = time.perf_counter()
        value = function(*args, **kwargs)
        endTime = time.perf_counter()

        runTime = endTime - startTime
        print(f"Finished {function.__name__!r} in {runTime:.4f} secs")

        return value

    return wrapperTimer




'''
	(ii) In the regular code, just specify the wrapper around any method

'''
@timeFunction
def addNextSessionPurchaseColumn(self):
	"""
		Adds a boolean column to specify if a purchase
		occurs in the next session by given cookie ID.
		Must be run after `addActionCountcolumns()` function.
	"""
	if 'A08_count' in self.df_sessions:
		self.df_sessions['purchase_next_session'] = 0
		self.df_sessions = self.df_sessions.sort_values(['cookie_id', 'startTime'], ascending=[True, True]).reset_index()
		self.df_sessions.apply(lambda row: self.nextSessionPurchases(row), axis=1)

		# Replace NaNs in `purchase_next_session` with 0
		self.df_sessions['purchase_next_session'] = self.df_sessions['purchase_next_session'].fillna(0)
		return self.df_sessions

	else:
		raise KeyError('Column `A08_count` does not exist in sessions dataframe. Ensure `addActionCountcolumns()` has been run first.')
		return self.df_sessions
