from img_to_vec import Img2Vec
from PIL import Image

# Initialize Img2Vec with GPU
img2vec = Img2Vec(cuda=False)

# Read in an image (rgb format)
this_file_path = '/Volumes/GoogleDrive/My Drive/DataScience/lighting talk/Mateo_badass.jpg'
img = Image.open(this_file_path)
# Get a vector from img2vec, returned as a torch FloatTensor
vec = img2vec.get_vec(img, tensor=True)
embeddins = vec.cpu().detach().numpy().squeeze()
# Or submit a list
# vectors = img2vec.get_vec(list_of_PIL_images)