movie_data = IncludeFile("movie_data",
                         help="The path to a movie metadata file.",
                         default=script_path('movies.csv'))


# Compute genre specific movies and a bonus movie in parallel.
self.next(self.bonus_movie, self.genre_movies)




python3 ./metaflow-tutorials/01-playlist/playlist.py run



Access previous runs

run = Flow('PlayListFlow').latest_successful_run
print("Using run: %s" % str(run))



# Automatically run in parallel

# We want to compute some statistics for each genre. The 'foreach'
# keyword argument allows us to compute the statistics for each genre in
# parallel (i.e. a fan-out).
self.next(self.compute_statistics, foreach='genres')



# Version-specific libraries
from metaflow import conda

@conda(libraries={"pandas": "0.25.3"})
@step
def flatten_candidates(self):
    """
    A step for metaflow to flatten the JSON from the candidates.

    """
    # get the candidates
    flattened_candidates = []
    all_possible_names = []
    col_names = []




# the most common use case for metaflow.S3 is to store auxiliary data in a Metaflow flow.
# For example: a file with the accuracy of the model...
from metaflow import FlowSpec, step, S3
import json

class S3DemoFlow(FlowSpec):

    @step
    def start(self):
        with S3(run=self) as s3:
            message = json.dumps({'message': 'hello world!'})
            url = s3.put('example_object', message)
            print("Message saved at", url)
        self.next(self.end)

    @step
    def end(self):
        with S3(run=self) as s3:
            s3obj = s3.get('example_object')
            print("Object found at", s3obj.url)
            print("Message:", json.loads(s3obj.text))

if __name__ == '__main__':
    S3DemoFlow()


# Catch and retry decorators:
  @catch(var='compute_failed')
  @retry(times=1)
  @step
  def compute_statistics(self):
    """Compute statistics for a single genre. Run in cloud"""
    self.genre = self.input
    # TODO: Computing statistics for a genre
    self.next(self.join)