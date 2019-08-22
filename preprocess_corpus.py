# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:13:14 2019

@author: raj99
"""
import os
import re
puncts = ['..',' ..',',,',',.',', ,','.,',' ,','\f', '|', ';','$', '&', '/', ', ,',' ,',',,','.,',', .',
 '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}','. .',',.',', .'
 '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', 'Â', '█', '½', 'à', '…', 
 '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', 
 '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', 
 '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩',
 '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', '(',')',]
#final corpus preprocessing also for class label doc preprocessing

def clean_text(name, fcorpus):
    with open(name, encoding = "utf-8", mode="r") as infile, open(fcorpus,"a",encoding = "utf-8") as fout:    
        lines = infile.readlines()
        counter =0
        for line in lines:
            print("current line",line)
            
            text = str(line)
            for punct in puncts:
                if punct in text:
                    text = text.replace(punct,'')
                    print(text)
            
            text = re.sub(r'[^a-zA-Z0-9\s\n,.]',r'',text)
            text = re.sub(r'[0-9]{2,}','',text) #to remove 2 and above numbers
#           text = re.sub(r'[0-9]{5,}','',text) #to remove 5 and above numbers
#           text = re.sub(r'(?:\b|-)([1-9]{1,3}[0]?|100)\b','',text) #to remove numbers except years
#           text = re.sub(r'([A-Za-z]+[\d@]+[\w@]*|[\d@]+[A-Za-z]+[\w@]*','',text) # to remove number and word together                       
            text = re.sub(r'(?<=\b\w)\s(?=\w\b)',r'',text)  #remove spaces b/w chars of a word
            text = re.sub(r"\b(\w)\1+\b}",r"",text) #to remove repeated same char in word boundary
            text = re.sub(r'(,\s){2,}',r'',text)  ##remove more than one "," at the sameplace           
            text = re.sub(r'(\.){2,}',r'',text)  #remove more than one . at the sameplace
            text = text.replace("\"",'')
            text = text.replace(',',' ')
            text = text.replace(" .",' ')
            
            text = re.sub(r'\s+',' ',text).strip()  #remove more than one space to single
            text = text.strip()
            text = text.lower()
            counter = counter+1
#            print("text to write:",text)
            print("line number", counter)
            fout.write(text)
            fout.write("\n")
                    
corpa_name = 'F:\Data\Engupdate_ap1\Text_corpus_Mar31.txt'
fcorpus = 'F:\Data\Text_corpus_Mar31NATO.txt'
#corpa_name = 'F:\\Data\\classlabel.txt'
#fcorpus = 'F:\\Data\\NATO_classlabel1.txt'

clean_text(corpa_name, fcorpus)