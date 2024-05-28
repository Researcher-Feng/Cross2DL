import re

import nltk
from nltk import word_tokenize as wt
from code_utils.clean import easy_clean_txt


code = r'a helper function for expm_2009 .'
code = re.sub('[0-9]', ' num ', code)
cl_code = re.sub('[^a-z]', ' ', code.lower())
cl_code = cl_code + ' .'
print(cl_code)
code = r'''build a datasetcollection with populated datasetcollectionelement objects corresponding to the {/tgt/sj/jjj}[]()87&*^ 4 dataset instances or throw exception if this is not a valid collection of the specified type .'''
# code = r'''convert the provided host to the format in t/proc/net/tcp* /proc/net/tcp uses little-endian four byte hex for ipv4 /proc/net/tcp6 uses little-endian per 4b word for ipv6 566 args: a/host: string with either hostname .'''
clean = easy_clean_txt(code, 'desc', 'split', True, False)
print(clean)
clean = easy_clean_txt(code, 'desc', 'sub', True, False)
print(clean)

wnl = nltk.stem.WordNetLemmatizer()
word = 'exception'
# lemmatized = wnl.lemmatize(word)
lemmatized = wnl.lemmatize(word, pos='v')
print(lemmatized)
print(wt(code))
import warnings
warnings.filterwarnings('ignore')
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp('exceptions')
print(doc[0].lemma_)
from difflib import SequenceMatcher
