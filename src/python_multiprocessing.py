# Mutiprocessing. use MP with a function that accepts several arguments, being the 1st one 
# the ONLY one that changes
	import multiprocessing as mp
	from functools import partial

	def worker(a,b,c,d):
		"""thread worker function"""
		print('A -> {},B -> {},C -> {},D -> {}'.format(a,b,c,d))
		return a-100

	cores      = mp.cpu_count() 
	maxCores   = cores-1
	pool       = mp.Pool(maxCores)
	# use partial to fix the arguments that don't change
	worker_a = partial(worker, b='b',c='c',d='d')
	dataChunks = range(10000)
	results = pool.map(worker_a, dataChunks)
	pool.close()
	pool.join()



# Mutiprocessing with DATAFRAMES
	import multiprocessing as mp
	from functools import partial

	def parallelSum(df,b,c,d):
		a   = df.TEST.sum()
		dfA = pd.DataFrame(data=[{'results': a}])
		print('A -> {},B -> {},C -> {},D -> {}'.format(a,b,c,d))
		return dfA

	df = pd.DataFrame(np.random.random_sample((50000,1)), columns=['TEST'])
	df_split = np.array_split(df, 100)

	cores      = mp.cpu_count() 
	maxCores   = cores-1
	pool       = mp.Pool(maxCores)
	# use partial to fix the arguments that don't change
	worker_a = partial(parallelSum, b='b',c='c',d='d')
	results = pool.map(worker_a, df_split)
	pool.close()
	pool.join()
	# The results are a list of dataframes
	extResults  = pd.DataFrame();
	for thisDF in results:
		extResults  = extResults.append(thisDF, ignore_index=True);