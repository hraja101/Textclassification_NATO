# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 00:47:34 2019

@author: raj99
"""
import os
import re

puncts = ['..',' ..',',,',',.',', ,','.,',' ,','\f', '|', ';','$', '&', '/', 
 '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}',
 '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', 'Â', '█', '½', 'à', '…', 
 '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', 
 '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', 
 '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩',
 '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def lines_multiple_extract(textdirect, outdir,finaldir,corpus_name):
    for filename in os.listdir(textdirect): #iterate through all text files in input directory
        in_file = textdirect + filename #gets the filename with diectory
        with open(in_file, encoding = "ISO-8859-1") as fin: #input file form preprocess 
            lines = fin.readlines()
            counter = len(lines)
            k = int(counter//50)
            file_count = 0  
            if (in_file.find("ENG") !=-1 and counter >=6): #only eng texts with number of lines>5 
                
                out_file = outdir + filename #output directory of text file
                print(out_file)
                print("lenght of the file:",counter) 
                
                final_file = finaldir + filename #final directory  of text files with all preprocessing 
                
                ######### start the writing of files here #######
                
                with open(out_file, encoding = "ISO-8859-1",mode= 'w+') as fout, \
                open(final_file, encoding = "ISO-8859-1",mode= 'w+') as final, \
                open(corpus_name, encoding = "utf-8",mode= 'a') as corpus:                           
                    if counter<=100:
#                       print("lines test:",lines[:6])
                        fout.writelines(lines[7:])
                        print("done if split")
                        fout.seek(0)
                        new_lines = fout.readlines()
                        for line in new_lines:
                            if line.strip() and len(line)>=3:
                                final.write(line[:-1])
               
                    elif counter>100 and counter<=1500:
                        fout.writelines(lines[8:40] + lines[counter//2:counter//2 + 30] + lines[-30:])
                        print("done elif split")
                        fout.seek(0)
                        new_lines = fout.readlines()
                        for line in new_lines:
#                           print("elif loop")
                            if line.strip() and len(line)>=3:
                                final.write(line[:-1])
                    
                    elif counter>1500 and counter<=2500:
                        fout.writelines(lines[8:k] + lines[counter//2:counter//2 + 25] + lines[-k:-3])
                        print("done elif split")
                        fout.seek(0)
                        new_lines = fout.readlines()
                        for line in new_lines:
#                           print("elif loop")
                            if line.strip() and len(line)>=3:
                                final.write(line[:-1])
                            
                    else:
                        fout.writelines(lines[10:k//2] + lines[counter//2:counter//2 + 25] + lines[-k//2:-5])
                        print("done else split")
                        fout.seek(0)
                        new_lines = fout.readlines()
                        for line in new_lines:
#                           print("else loop")
                            if line.strip() and len(line)>=3:
                                final.write(line[:-1])
                    final.truncate()
                    final.write("\n")
                    file_count = file_count + 1
                    final.seek(0) # got to first line of file of the current file object
#                    print("lines for label:",lines[:10])
                    
                #read from the files to create corpus
                
                    getlines = final.read() #read all the content
                    print("file:",len(getlines),"is of length:", final_file)            
#                    getstr_file = str(getlines)
                    print(getlines)
#                    
#                    for punct in puncts:
#                        if punct in getstr_file:                     

#                            getstr_file = re.sub(r'[^a-zA-Z0-09\s\n,.]',r'',getstr_file)  #removes all except char,num,space, dot and comma
#                           getstr_file = re.sub(r'[0-9]{5,}','',getstr_file)
#                            getstr_file = re.sub(r'[0-9]{2,}','',getstr_file) #removes all numbers above 2 length
#                           getstr_file = re.sub(r'(?:\b|-)([1-9]{1,3}[0]?|100)\b','',getstr_file) #removes no.s except years 
#                            getstr_file = re.sub(r'(?<=\b\w)\s(?=\w\b)',r'',getstr_file) #removes spaces between chars
#                            getstr_file = getstr_file.replace(punct,'') #removes all punct
#                            getstr_file = getstr_file.replace(' .','')
#                            getstr_file = getstr_file.replace(',,','')
#                            getstr_file = getstr_file.replace("  ",' ') #removes double space to single
#                            getstr_file = getstr_file.strip()
#                            getstr_file = getstr_file.lower()
                    print("writing corpa from file:",final_file)
                    corpus.write(getlines)
                    
#                    corpus.write("\n")
                print("count",file_count)
                
textdirect = "F:/Data/PreprocessText/"
outdir = "F:/Data/EngpreprocessText_ap1/"
finaldir = "F:/Data/Engupdate_ap1/"
corpus_name = "F:/Data/Engupdate_ap1/Text_corpus_Mar31.txt"

lines_multiple_extract(textdirect, outdir,finaldir,corpus_name)

