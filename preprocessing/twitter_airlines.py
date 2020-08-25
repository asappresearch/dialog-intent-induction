import re

import nltk


class Preprocessor(object):
    def __init__(self):
        self.twitter_at_re = re.compile('@[a-zA-Z1-90]+')
        self.initials_re = re.compile('\^[A-Z][A-Z]([^A-Za-z]|$)')
        self.star_initials_re = re.compile('\*[A-Z]+$')
        self.money_re = re.compile('\$[1-90]+([-.][1-90]+)?')
        self.number_re = re.compile('[1-90]+([-.: ()][1-90]+)?')
        self.airport_code_re = re.compile('([^a-zA-Z])[A-Z][A-Z][A-Z]([^a-zA-Z])')
        self.tag_re = re.compile('([^a-zA-Z])#[a-zA-Z]+')
        self.url_re = re.compile('http[s]?:(//)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  # from mleng data_preprocess.py

    def __call__(self, text, info_dict=None):
        is_valid = True
        if 'DM' in text:
            is_valid = False
        if 'https://t.co' in text:
            is_valid = False

        target_company = info_dict['target_company']
        cust_twitter_id = info_dict['cust_twitter_id']
        text = text.replace('\n', ' ').encode('ascii', 'ignore').decode('ascii')
        text = text.replace('@' + target_company, ' __company__ ').replace('@' + cust_twitter_id, ' __cust__ ')
        text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
        text = self.twitter_at_re.sub('__twitter_at__', text)
        text = self.initials_re.sub('__initials__', text)
        text = self.star_initials_re.sub('__initials__', text)
        text = self.money_re.sub(' __money__ ', text)
        text = self.number_re.sub(' __num__ ', text)
        text = self.airport_code_re.sub('\g<1> __airport_code__ \g<2>', text)
        text = self.tag_re.sub('\g<1> __twitter_tag__ ', text)
        text = text.casefold()
        text = self.url_re.sub(' __url__ ', text)
        text = ' '.join(nltk.word_tokenize(text))

        return is_valid, text
