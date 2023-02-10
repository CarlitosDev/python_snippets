



# Apache Tika nails it...
# https://en.wikipedia.org/wiki/Apache_Tika
from tika import parser
raw = parser.from_file(local_path)
print(raw['content'])



