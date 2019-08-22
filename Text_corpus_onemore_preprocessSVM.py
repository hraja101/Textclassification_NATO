# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:18:42 2019

@author: raj99
"""
import os
import re
puncts = ['..',' ..',',,',',.',', ,','.,',' ,','\f']
#final corpus preprocessing also for class label doc preprocessing
puncts = ['..',' ..',',,',',.',', ,','.,',' ,','\f',':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]


def clean_text(name, fcorpus):
    with open(name, encoding = "utf-8", mode="r") as infile, open(fcorpus,"a",encoding = "utf-8") as fout:    
       lines = infile.readlines()
       for line in lines:
            text = str(line)
            text = text.replace(" .",'.')
            text = text.replace("..",'.')
            print("writing line")
            fout.write(text)
                    

#corpa_name = 'F:\Data\TextDoc_Corpus_NATO.txt'
#fcorpus = 'F:\Data\TextDoc_Corpus_NATO_DLpreprocess.txt'
#corpa_name = 'F:\\Data\\classlabel.txt'
#fcorpus = 'F:\\Data\\NATO_classlabel1.txt'

clean_text(corpa_name, fcorpus)