benchmark_serialisation_formats_chatGPT.py


import time
import pickle
import msgpack

def dict_benchmark(dct):
    """
    Benchmark the serialization and deserialization time for a dictionary using pickle and msgpack
    """

    # Benchmark pickle
    start_time = time.time()
    pickled_data = pickle.dumps(dct)
    end_time = time.time()
    print(f'Pickle dump time: {end_time - start_time}')

    start_time = time.time()
    pickle.loads(pickled_data)
    end_time = time.time()
    print(f'Pickle load time: {end_time - start_time}')

    # Benchmark msgpack
    start_time = time.time()
    msgpacked_data = msgpack.packb(dct)
    end_time = time.time()
    print(f'Msgpack dump time: {end_time - start_time}')

    start_time = time.time()
    msgpack.unpackb(msgpacked_data)
    end_time = time.time()
    print(f'Msgpack load time: {end_time - start_time}')

dct = {'a': 1, 'b': 2, 'c': {'d': 3, 'e': 4}}
dict_benchmark(dct)


