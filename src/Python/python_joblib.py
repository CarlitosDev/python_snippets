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








'''
    Example with the Lumiere tools
'''
from joblib import Parallel, delayed
## Get the list of input files
bucket_contents = awsu.get_buckets_contentsV2(input_bucket, input_bucket_key)
s3_path = ['s3://' + input_bucket + '/' + iFile \
  for iFile in bucket_contents if video_extension in iFile]


filename_no_ext = [iFile.replace('.' + video_extension,'').split('/')[-1] \
    for iFile in s3_path]


df_videos_to_analyse = pd.DataFrame({'filename_no_ext': filename_no_ext,
's3_path':s3_path})


# split into num_processes
num_processes = 8
videoList = np.array_split(df_videos_to_analyse, num_processes, axis=0)

# Transcribe
transcribed_file_list = [iFile.replace('_transcribe.json','') \
    for iFile in already_analysed_files if 'transcribe' in iFile]

if df_videos_to_analyse.shape[0] > 1:
  
  queryStart = time.time()
  r = Parallel(n_jobs=-1)(delayed(lmu.rerun_transcribe)(file_selection, \
    output_bucket, transcribed_file_list) for file_selection in videoList)
  queryElapsed = time.time() - queryStart
  print(f'...arranging videos done in {queryElapsed:.2f} sec!')
else:
  lmu.rerun_transcribe(df_videos_to_analyse, output_bucket, transcribed_file_list)