import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
class TextCleaner:
    
    def __init__(self, text, text_form, stemming, lemmat):
        
        self.text = text
        self.text_form = text_form
        self.stemming = stemming
        self.lemmat = lemmat

    def lower(self):
        if (self.text_form == 'One-All'):
            if type(self.text)==list:
                low_text = self.text[0].lower()
            elif (type(self.text)==str):
                low_text = self.text.lower()
        if (self.text_form == 'List'):
            low_text = []
            for t in self.text:
                low_text.append(t.lower())
        return (low_text)
    
    #punctuation remover

    def Punctuator(self, words):
        punctuations = '''~`@#$%^&*()_-+=/*,-.[]""{};:’''""“” £<>¢?\—|1234567890x92\n'''
        if (self.text_form=='One-All'):
            if (type(words)==list):
                word_join = []
                for t in words[0].split():
                    for p in punctuations:
                        if (p in t):
                            t = t.replace(p, '')
                    word_join.append(t)
            elif (type(words)==str):
                word_join = []
                for t in words.split():
                    for p in punctuations:
                        if (p in t):
                            t = t.replace(p, '')
                    word_join.append(t)    
        if (self.text_form == 'List'):
            word_join = []
            for ew in words:
                each_len = []
                for w in ew.split():
                    for p in punctuations:
                        if (p in w):
                            w = w.replace(p, '')
                    each_len.append(w)   
                word_join.append(' '.join(each_len))             
        return(word_join)
    
    #stop words remove

    def Stopwords(self, tex):
        if (self.text_form == 'One-All'):
            if (type(tex)==list):
                stop_text = []
                text_stop = [i for i in tex[0].split() if i not in stopwords.words('english')]
                stop_text.append(' '.join(text_stop))
            elif (type(tex)==str):
                stop_text = []
                text_stop = [i for i in tex.split() if i not in stopwords.words('english')]
                stop_text.append(' '.join(text_stop))
        if (self.text_form =='List'):
            stop_text = []
            for t in tex:
                text_stop = ' '.join([i for i in t.split() if i not in stopwords.words('english')])
                stop_text.append(text_stop)
        return(stop_text)
    
    
    def WordNorm(self, stp):
        word_norm = []
        if (self.stemming):
            stm = PorterStemmer()
            if(self.text_form=='One-All'):
                if(type(stp)==list):
                    stm_words = [stm.stem(w) for w in stp[0].split()]
                    word_norm.append(' '.join(stm_words))
                if (type(stp)==str):
                    stm_words = [stm.stem(w) for w in stp.split()]
                    word_norm.append(' '.join(stm_words))
            if (self.text_form == 'List'):
                each_norm = []
                for t in stp:
                    stm_words = [stm.stem(w) for w in t.split()]
                    word_norm.append(' '.join(stm_words))
                
        if (self.lemmat):
            lem = WordNetLemmatizer()
            if(self.text_form=='One-All'):
                if(type(stp)==list):
                    lem_words = [lem.lemmatize(w) for w in stp[0].split()]
                    word_norm.append(' '.join(lem_words))
                if (type(text)==str):
                    lem_words = [lem.lemmatize(w) for w in stp.split()]
                    word_norm.append(' '.join(lem_words))
            if (self.text_form == 'List'):
                each_norm = []
                for t in stp:
                    lem_words = [lem.lemmatize(w) for w in t.split()]
                    word_norm.append(' '.join(lem_words))             
        return (word_norm)
    
    def clean(self):
        words = self.lower()
        pl = self.Punctuator(words)
        stop_tex = self.Stopwords(pl)
        norm = self.WordNorm(stop_tex)
        return(norm)