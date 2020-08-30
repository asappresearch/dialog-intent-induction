import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup


REMOVE_STOP = False
QUESTION_WORDS = ["what", "when", "where", "why", "how", "who"]
STOPWORDS = set(stopwords.words("english"))
for w in QUESTION_WORDS:
    STOPWORDS.remove(w)

# replace URL with a special token *=URL=*
URL_PATTERN = r'(?:(?:https?|ftp)://)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

URLREG = re.compile(URL_PATTERN)


def inner_preprocess_text_from_tao(text, max_len):
    # remove non-ascii chars
    text = text.encode("utf8").decode("ascii", "ignore")

    # remove URLs
    text = URLREG.sub('*=URL=*', text)

    text = text.casefold()

    # tokenize, filter and truncate
    words = word_tokenize(text)
    if REMOVE_STOP:
        words = [x for x in words if x not in STOPWORDS]
    if max_len > 0:
        words = words[:max_len]
    return u' '.join(words)


def preprocess_from_tao(text, max_len):
    """
    adapted from Tao's code for http://aclweb.org/anthology/D18-1131
    """
    body_soup = BeautifulSoup(text, "lxml")

    # remove code
    [x.extract() for x in body_soup.findAll('code')]

    # also remove pre
    [x.extract() for x in body_soup.findAll('pre')]

    # remove "Possible Duplicate" section
    blk = body_soup.blockquote
    if blk and blk.text.strip().startswith("Possible Duplicate:"):
        blk.decompose()
    body_cleaned = inner_preprocess_text_from_tao(body_soup.text, max_len=max_len)
    assert "Possible Duplicate:" not in body_cleaned

    assert "\n" not in body_cleaned
    return body_cleaned


class Preprocessor(object):
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, text):
        return True, preprocess_from_tao(text, max_len=self.max_len)
