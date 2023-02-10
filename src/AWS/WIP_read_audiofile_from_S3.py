WIP_read_audiofile_from_S3.py


# This is one solution....
bytes_object = evc_utils.read_bytes_from_S3(bucket_name, key_path, s3_client)
# (A)
import os, tempfile
tmp = tempfile.NamedTemporaryFile(delete=False)
try:
    print(tmp.name)
    tmp.write(bytes_object)
    y, sr = librosa.load(tmp.name)
finally:
    tmp.close()
    os.unlink(tmp.name)

# (B)
import tempfile
with tempfile.NamedTemporaryFile() as tmp:
    print(tmp.name)
    tmp.write(bytes_object)
    y_signal, fs = librosa.load(tmp.name)