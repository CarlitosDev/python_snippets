from joblib import Parallel, delayed
class square_class_v2:
    def square_int(self, i):
        return i * i
     
    def run(self, num):
        results = []
        results = Parallel(n_jobs= -1, backend="threading")\
            (delayed(self.unwrap_self)(i) for i in zip([self]*len(num), num))
        print(results)
    @staticmethod
    def unwrap_self(arg, **kwarg):
        return square_class_v2.square_int(*arg, **kwarg)

square_int = square_class_v2()
square_int.run(num = range(10))