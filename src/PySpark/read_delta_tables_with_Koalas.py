

%python
import databricks.koalas as ks
df = ks.read_delta("/englishlive/unitCompleted/session/silver")
df.tail()