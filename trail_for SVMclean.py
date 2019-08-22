# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:34:11 2019

@author: raj99
"""
path_Corpus = 'F:/Data/TextDL_Learn/TextCorpus/'
corpa_name = path_Corpus + 'Text_corpus_Mar31NATO.txt'
label_names = path_Corpus + 'Label_numbersNATO.txt'

import re
import os

def clean_str(text):
    text = text.replace(".", "")
    text = re.sub(r'[0-9,]',r'',text)
    text = re.sub(r'\s+',' ',text).strip()
    return text

with open(corpa_name) as file_doc:
    text_list = file_doc.readlines() #get corpus --> list 
    text_list = [clean_str(x) for x in text_list]
with open(label_names) as file_label:
    labels_list = file_label.readlines() #get corpus labels --> list
    
print(text_list[5:7])