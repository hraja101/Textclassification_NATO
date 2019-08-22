# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 20:14:07 2019

@author: raj99
"""
#creates a label file 
import os
import re
Excep = ['secretary','secretory','secretariat','secretaries','secretarie' ]

def get_label(name, fcorpus, actual_label):
    with open(name, encoding = "utf-8", mode="r") as infile, \
    open(fcorpus,"a",encoding = "utf-8") as fout, open(actual_label,"a",encoding = "utf-8") as file_f:    
        lines = infile.readlines()
        for line in lines:
            text = str(line)
            text = text.lower()
            print(text)
            count = 0
            if re.search('(?:restrict)(..)',text):
                match = re.search('(?:restrict)(..)',text)
                label = 'restricted'
                match = match.group(0)
                
            elif re.search('(?:confidential)()',text):
                match = re.search('(?:confidential)()',text)
                label = 'confidential'
                match = match.group(0)
                
            elif re.search('(?:secret)(.....)',text) :
                for ex in Excep:
                    if ex in text:
                        count = count+1                    
                print("value:",count)
                    
                if(count == 0):
                    match = re.search('(?:secret)(.....)',text)
                    label = 'secret'
                    match = match.group(0)
                
            else:
                label = 'unclassified'
                match = 'unclassified'
            
            fout.write(label)
            fout.write('\n')
            print("matched:",match)
            file_f.write(match)
            file_f.write('\n')
                     
#corpa_name = 'F:\Data\TextDoc_Corpus_NATO.txt'
#fcorpus = 'F:\Data\TextDoc_Corpus_NATO_DLpreprocess.txt'
corpa_name = 'F:\\Data\\NATO_classlabel1.txt'
fcorpus = 'F:\\Data\\NATO_classlabe3.txt'
actual_label = 'F:\\Data\\NATO_classlabe4.txt'

get_label(corpa_name, fcorpus, actual_label)