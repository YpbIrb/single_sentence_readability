import os
import time
from spacy.tokens import Doc
from spacy.vocab import Vocab
import pandas as pd

from feature_extractor import FeatureExtractor
from text_collnu_converter import Text2CollnuConverter
from constants import ssr_constants


class MainProcessor:

    def __init__(self, _nlp):
        self.nlp = _nlp

    
    #Велосипед для того, чтобы в таблице, там где tag и dep _num были инты, а не float
    #!! ВАЖНО: при внесение изменений в порядок столбцов в датафрейме, возможно необходимо будет поменять константу снизу
    def _fillNan_and_remove_floats(self, features):
        features = features.fillna(int(0))
        start_of_non_float_values = 10
        columns_list = list(features.iloc[:, start_of_non_float_values:])
        features[columns_list] = features[columns_list].astype(int)
        return features
    
    def get_doc_features(self, _filename, doc, _grade):
        extractor = FeatureExtractor()

        time_extract_start = time.time()
        features = extractor.get_doc_features_df(doc)
        time_extract_end = time.time()
        extraction_time = time_extract_end - time_extract_start

        # add grade and text file to features
        features.insert(0, "Grade", int(_grade), allow_duplicates=False)
        features.insert(0, "Filename", _filename, allow_duplicates=False)
        
        features = self._fillNan_and_remove_floats(features)

        print("features extraction time : " + str(extraction_time))
        print("features extraction time per sentence : " + str(extraction_time / len(list(doc.sents))))
        return features

    def process_saved_doc(self, filepath, grade):
        doc = Doc(Vocab()).from_disk(filepath)

        head, tail = os.path.split(filepath)
        filename = tail.split('.')[0]

        features = self.get_doc_features(filename, doc, grade)
             
        features.to_csv(path_or_buf=ssr_constants.ROOT_FOLDER_PATH + "/features_dfs/" + filename + "_features", index=False)

        return features

    def process_raw_text(self, filepath, grade):
        doc_coverter = Text2CollnuConverter(self.nlp)
        doc_path, doc = doc_coverter.get_doc_from_text(filepath)

        head, tail = os.path.split(filepath)
        filename = tail.split('.')[0]

        features = self.get_doc_features(filename, doc, grade)
        features.to_csv(path_or_buf=ssr_constants.ROOT_FOLDER_PATH + "/features_dfs/" + filename + "_features", index=False)

        return doc, features

    def process_docs_list(self, listpath):
        path_dict = {}
        with open(listpath) as f:
            for line in f:
                key, val = line.split()
                path_dict[str(key)] = int(val)
        features = pd.DataFrame()

        for i in path_dict:
            print("Extracting features from " + i)
            curr_features = self.process_saved_doc(i, path_dict[i])
            features = pd.concat([features, curr_features],  ignore_index = True)
            
        features = features.fillna(0)   
        
        
        features = self._fillNan_and_remove_floats(features)
        #start_of_non_float_values = 10
        #columns_list = list(features.iloc[:, start_of_non_float_values:])
        #features[columns_list] = features[columns_list].astype(int)    
        
        return features

    def process_raw_list(self, listpath):
        doc_coverter = Text2CollnuConverter(self.nlp)
        doc_list_path, docs = doc_coverter.convert_mult_docs(listpath)
        features = self.process_docs_list(doc_list_path)
        return features

    def load_dataframes(self, listpath):
        features = pd.DataFrame()
        file = open(listpath, 'r')
        Lines = file.read().splitlines()
        #Lines = file.readlines()
 
        for line in Lines:
            df = pd.read_csv(line)
            features = pd.concat([features, df], ignore_index=True)
        
        features = self._fillNan_and_remove_floats(features)
        
        return features
        

if __name__ == '__main__':
    print("Hello")
