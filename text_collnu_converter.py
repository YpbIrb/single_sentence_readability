import stanza
import spacy_stanza
from spacy import displacy
import time
import pandas as pd
from spacy_conll import init_parser
from spacy.tokens import Doc
from spacy.vocab import Vocab
import os
from constants import ssr_constants


class Text2CollnuConverter:

    def __init__(self, _nlp):
        self.nlp = _nlp

        
    def _check_valid_line(self, sentence):
        if (sentence[0].islower()):
            #print("11111")
            #print(sentence)
            return False
        
        if (sentence[-3] == "т" and sentence[-4] == " "):
            #print("22222")
            #print(sentence)
            return False
        
        if (sentence[-3].isupper() and sentence[-4] == " "):
            #print("33333")
            #print(sentence)
            return False
        
        return True
        
    
    def preprocess_text_list(self, listpath):

        file_in = open(listpath, 'r')
        Lines = file_in.read().splitlines()
        file_in.close
        
        removed_lines_count = 0
        for line in Lines:
            curr_removed_lines = self.preprocess_text(line)
            removed_lines_count += curr_removed_lines
            
        return removed_lines_count
            
    
    def preprocess_text(self, filepath):
        
        file_in = open(filepath, "r", encoding='utf-8')
        #file_str = file_in.read()
        Lines = file_in.readlines()
        file_in.close()
        
        
        new_Lines = [x for x in Lines if self._check_valid_line(x) == True]
        
        removed_lines_count = len(Lines) - len(new_Lines)
        
        head, tail = os.path.split(filepath)
        filename = tail.split('.')[0]
        preprocessed_doc_path = ssr_constants.ROOT_FOLDER_PATH + "/preprocessed/" + filename + "_preprocessed"
        
        file_out = open(preprocessed_doc_path, "w", encoding='utf-8')
        file_out.writelines(new_Lines)
        file_out.close()
        
        return removed_lines_count
        
        
    def get_doc_from_text(self, filepath):

        file_in = open(filepath, "r", encoding='utf-8')
        file_str = file_in.read()
        file_in.close()

        print("Converting " + filepath)

        time_conv_start = time.time()
        doc = self.nlp(file_str)
        time_conv_end = time.time()
        converting_time = time_conv_end - time_conv_start

        head, tail = os.path.split(filepath)
        filename = tail.split('.')[0]

        # save doc in file
        doc_path = ssr_constants.ROOT_FOLDER_PATH + "/docs/" + filename + "_doc"
        doc.to_disk(doc_path)

        print("colln converting time : " + str(converting_time))
        print("colln converting time per_sentence : " + str(converting_time / len(list(doc.sents))))

        return doc_path, doc

    def convert_mult_docs(self, listpath):
        paths_list = []
        grades_list = []
        with open(listpath) as f:
            for line in f:
                path, grade = line.split()
                paths_list.append(path)
                grades_list.append(int(grade))

        docs = []
        docs_paths = []

        for i in range(len(paths_list)):
            doc_path, doc = self.get_doc_from_text(paths_list[i])
            docs.append(doc)
            docs_paths.append(doc_path)

        doc_list_path = listpath + "_docs"
        f = open(doc_list_path, "w")
        for i in range(len(docs_paths)):
            f.write(docs_paths[i] + " " + grades_list[i] + "\n")

        f.close()

        return doc_list_path, docs
