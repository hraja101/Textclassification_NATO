# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:43:33 2019

@author: raj99
"""
import os
import re

def lines_multiple(textdirect, outdir):
    puncts = ['  ',':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√',]
 
    """Write the file, removing empty lines and lines that contain only whitespace."""
    for filename in os.listdir(textdirect): #iterate through all text files in input directory
        textfile_name = textdirect + filename #gets the filename with diectory
        if (textfile_name.find("ENG") !=-1): #only eng texts
            print(textfile_name)
            outfile_name = outdir + filename #output directory text file
            print(outfile_name)
            with open(textfile_name, encoding = "ISO-8859-1") as in_file, open(outfile_name, encoding = "ISO-8859-1",mode= 'w+') as out_file:
                lines = in_file.readlines()
                counter = len(lines)
                print("infile:",counter)
#                print(lines)
                for line in lines:             
                    if line.strip() and len(line)>=3:
                        if counter < 100 and counter>5:
#                            new_line = re.sub(r'[^a-zA-Z0-9 ,.]',r'',str(line))
#                            out_file.write(line[:-1].replace(pun for pun in puncts if pun in str(line),' ')) 
                            out_file.write(line[:-1])
                        elif counter>100 and counter<400 :
#                            new_line1 = re.sub(r'[^a-zA-Z0-9 ,.]',r'',str(line))
#                            out_file.write(line[70:-1].replace(pun for pun in puncts if pun in str(line),' ')+ line[-20:-1].replace(pun for pun in puncts if pun in str(line),' '))
                            out_file.write(line[:-1] + line[:-1])
                out_file.truncate()
                out_file.write("\n")

textdirect = "F:/Data/PreprocessText/"
outdir = "F:/Data/EngpreprocessText/"
lines_multiple(textdirect, outdir)
#def clean_text(outdir):
#    for fname in os.listdir(outdir):
#        clean_name = outdir+fname
#        with open(clean_name) as in_file, open(clean_name, mode= 'r+') as out_file:
#            pattern = r'[^a-zA-z0-9\s]'
#            text = re.sub(pattern,'',str(in_file))
#            text = re.sub('  ',' ',str(in_file))
#            out_file.write(text)
            

#clean_text(outdir)