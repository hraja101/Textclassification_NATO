# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 23:39:25 2019

@author: raj99
"""
#gets a class label from corpus and creates a corpus
import os

def lines_multiple_extract(textdirect, outdir,label_dir,label_file):
    for filename in os.listdir(textdirect): #iterate through all text files in input directory
        in_file = textdirect + filename #gets the filename with diectory

        with open(in_file, encoding = "ISO-8859-1") as fin:
            lines = fin.readlines()
            counter = len(lines)
            if (in_file.find("ENG") !=-1 and counter>=6): #only eng texts
#                print(in_file)
                out_file = outdir + filename #output directory text file
                class_file = label_dir + filename      
                with open(out_file, encoding = "ISO-8859-1",mode= 'w+') as fout,\
                open(class_file, encoding = "ISO-8859-1",mode= 'w+') as cl_lb, \
                open(label_file, encoding = "utf-8",mode= 'a') as label:

                    print(counter)
                    fout.writelines(lines[:25])
                    print("done split")
                    fout.seek(0)
                    new_lines = fout.readlines()
#                    print(out_file, "lines:",new_lines)
                
                    for line in new_lines:
                        if line.strip() and len(line)>=3:
                            cl_lb.write(line[:-1])
                    cl_lb.truncate()
                    cl_lb.write("\n")
                    cl_lb.seek(0)
#                    print("label writing:",filename)
                    
                    getlines = cl_lb.read() #read all the contents  
                    print("now the lines:",getlines)
                    print("writing from:",class_file)
                    label.write(getlines)
#
                                        
textdirect = "F:/Data/PreprocessText/"
outdir = "F:/Data/labels/"
label_dir = "F:/Data/labels1/"
label_file = "F:/Data/labels/classlabel.txt"

lines_multiple_extract(textdirect, outdir,label_dir,label_file)
