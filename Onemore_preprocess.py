# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 10:12:41 2019

@author: raj99
"""

import re
s = 'chairmen and presidents of the north atlantic c ouncil gross national product at constant p ric es investment at \
constant prices. private consumption per capita at constant prices. private consumption and defense expenditures in \
european member inside front coverinside back coverco un tries iii jjjj kk nato aid to u\
and st at is t ic al tables1. the nato area 2.the nato commands 3.soviet expansion during and after world war ii\
4.population and area of natocoun tries. 5.the nato civil and military organization. 6.the principal committees \
of the counc il. 7.the nato international staffsecretariat'


def clean_text(name, fcorpus):
    with open(name, encoding = "utf-8", mode="r") as infile, open(fcorpus,"a",encoding = "utf-8") as fout:    
        lines = infile.readlines()
        
        counter =0
        for line in lines:
            text = str(line)
            print("current line",line)         
            text = re.sub(r"\b(\w)\1+\b",r"",text) 
            text = re.sub(r"[0-9]",r"",text)
            text = text.replace(" .",'')
            text = re.sub(r'\s+',' ',text).strip()  #remove more than one space to single
            text = text.strip()
            text = text.lower()
            counter = counter+1
#            print("text to write:",text)
            print("line number", counter)
            fout.write(text)
            fout.write("\n")
                    
corpa_name = 'F:\Data\TextDL_Learn\TextCorpus\DL_Text_corpus_Mar31NATO.txt'
fcorpus = 'F:\Data\TextDL_Learn\TextCorpus\DL_Text_corpus_NATOupdated.txt'
#corpa_name = 'F:\\Data\\classlabel.txt'
#fcorpus = 'F:\\Data\\NATO_classlabel1.txt'

clean_text(corpa_name, fcorpus)


print(s) 