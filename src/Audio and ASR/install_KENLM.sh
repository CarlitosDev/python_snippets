install_KENLM.sh

Language model


mkdir /Users/carlos.aguilar/Documents/EF_repos/kenLM
cd /Users/carlos.aguilar/Documents/EF_repos/kenLM

# wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz
git clone https://github.com/kpu/kenlm.git


brew install cmake boost eigen
brew install xz

liblzma-dev


mkdir kenlm/build && cd kenlm/build && cmake .. && make -j 4
# the executable functions have successfully been built under kenlm/build/bin/.
python3 setup.py install
python3 -c "import kenlm"




# Get one transcription to test kenLM

# prepare text for kenLM
import carlos_utils.file_utils as fu
path_to_json = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/06.06.2022/3ac46cad-ade4-42c0-85ba-9aade117b073/evc_API/3ac46cad-ade4-42c0-85ba-9aade117b073b.json'
json_data = fu.readJSONFile(path_to_json)
transcription = json_data['results']['transcripts'][0]['transcript']

path_to_txt = '/Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/06.06.2022/3ac46cad-ade4-42c0-85ba-9aade117b073/languageModels/transcription.txt'
fu.writeTextFile(transcription, path_to_txt)





# 
cd /Volumes/TheStorageSaver/29.12.2021-EducationFirst/EF_EVC_API_videos/adults_spaces/06.06.2022/3ac46cad-ade4-42c0-85ba-9aade117b073/languageModels/


/Users/carlos.aguilar/Documents/EF_repos/kenLM/kenlm/build/bin/lmplz -o 2 <"transcription.txt"> "2gram.arpa"
/Users/carlos.aguilar/Documents/EF_repos/kenLM/kenlm/build/bin/lmplz -o 3 <"transcription.txt"> "3gram.arpa"


# Use patrickvonplaten's fixer (explained here https://huggingface.co/blog/wav2vec2-with-ngram)
# I can't remember how to fetch the file...
# wget https://github.com/patrickvonplaten/Wav2Vec2_PyCTCDecode/blob/main/fix_lm.py
wget https://raw.githubusercontent.com/patrickvonplaten/Wav2Vec2_PyCTCDecode/main/fix_lm.py
chmod +x fix_lm.py

./fix_lm.py --path_to_ngram 2gram.arpa --path_to_fixed 2gram_fixed.arpa
./fix_lm.py --path_to_ngram 3gram.arpa --path_to_fixed 3gram_fixed.arpa


head -n 10 3gram_fixed.arpa


# repo with some n-grams created with https://huggingface.co/edugp/kenlm

# I am going to download this model trained on Wikipedia English: 
# https://huggingface.co/edugp/kenlm/blob/main/wikipedia/en.arpa.bin