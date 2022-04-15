import spacy_stanza
from spacy_conll import init_parser
from spacy.tokens import Doc
from spacy.vocab import Vocab
import pandas as pd


class FeatureExtractor:

    def get_doc_features_df(self, doc):
        df = pd.DataFrame()
        for sentence in list(doc.sents):
            # print(sentence)
            features = self.get_sentence_features(sentence)
            # print(features)
            df_features = pd.DataFrame([features])
            
            #common_columns = df.columns.union(df_features.columns)

            #df1 = df.reindex(columns=common_columns, fill_value=0)
            #df2 = df_features.reindex(columns=common_columns, fill_value=0)
            
            df = pd.concat([df, df_features], ignore_index=True)
        return df

    #НЕ Важно: при изменении порядка, или добавлении новых фич, нужно проверять, что в main_processor не затирается 
    #информация после превращения флоатов в инты. Текущий столбец, с которого начинается превращение - 10
    def get_sentence_features(self, sentence):
        res = dict()
        res["Valid"] = 1
        res.update(self._get_tree_depth(sentence))
        res.update(self._get_sent_length(sentence))
        res.update(self._get_mean_distance(sentence))
        res.update(self._get_max_distance(sentence))
        res.update(self._get_n_children(sentence))
        res.update(self._get_punct_num(sentence))
        res.update(self._get_tags_count(sentence))
        res.update(self._get_dependencies_count(sentence))
        return res

    # Расчет глубины дерева
    # Done
    def _get_tree_depth(self, sentence):
        res = dict()
        res["tree_depth"] = self._tree_depth_dfs(sentence.root, 0)
        return res

    def _tree_depth_dfs(self, node, depth):
        if node.n_lefts + node.n_rights > 0:
            return max(self._tree_depth_dfs(child, depth + 1) for child in node.children)
        else:
            return depth

    #Расчет количества токенов(без учета пунктуации) в предложении
    def _get_sent_length(self, sentence):
        res = dict()
        res["sent_length"] = 0
        for token in sentence:
            if token.tag_ != "_SP" and token.tag_ != "PUNCT":
                res["sent_length"] += 1
           
        return res
        
    # Расчет количества вершин с определенным количеством потомков(без учета "корня" как вершины с 1 ребенком)
    # Done
    @staticmethod
    def _get_n_children(sentence):
        res = dict()
        res["leaf_num"] = 0
        res["1_child"] = 0
        res["2_child"] = 0
        res["3_child"] = 0
        res["4+_child"] = 0

        for token in sentence:
            if token.tag_ != "_SP":
                children_count = 0
                for child in token.children:
                    children_count += 1
                if children_count == 0:
                    res["leaf_num"] += 1
                elif children_count == 1:
                    res["1_child"] += 1
                elif children_count == 2:
                    res["2_child"] += 1
                elif children_count == 3:
                    res["3_child"] += 1
                elif children_count >= 4:
                    res["4+_child"] += 1

        # Вычитание для поправки на наличие "корневой" вершины
        res["1_child"] -= 1
        
        res["non_leaf_num"] = res["1_child"] + res["2_child"] + res["3_child"] + res["4+_child"]
        return res

    # Расчет количества знаков пунктуации
    @staticmethod
    def _get_punct_num(sentence):
        res = dict()
        res["n_punct"] = 0
        for token in sentence:
            if token.tag_ == "PUNCT":
                res["n_punct"] += 1

        return res

    # Расчет максимальной дистанции в предложении
    # дистанция между пунктуацией и её предками не считается, однако пунктуация вносит вклад в расстояние между словами
    @staticmethod
    def _get_max_distance(sentence):
        res = dict()
        res["max_dist"] = 0
        max_dist = 0
        for token in sentence:
            if token.tag_ != "PUNCT" and token.tag_ != "_SP":
                max_dist = max(abs(token.i - token.head.i), max_dist)

        res["max_dist"] = max_dist
        return res

    # Расчет средней дистанции в предложении
    # (дистанция между пунктуацией и её предками не считается, однако пунктуация вносит вклад в расстояние между словами
    # также не учитываются вершины с нулевой дистанцией)
    @staticmethod
    def _get_mean_distance(sentence):
        res = dict()
        res["mean_dist"] = 0
        sum_dist = 0
        non_punct_count = 0
        for token in sentence:
            if token.tag_ != "PUNCT" and token.tag_ != "_SP" and abs(token.i - token.head.i) != 0:
                non_punct_count += 1
                sum_dist += abs(token.i - token.head.i)

        if non_punct_count == 0:
            res["Valid"] = 0
        else:
            res["mean_dist"] = sum_dist / non_punct_count
        return res

    #Расчет количества всех видов тегов, без пунктуации
    @staticmethod
    def _get_tags_count(sentence):
        res = dict()
        for token in sentence:
            if token.tag_ != "_SP" and token.tag_ != "PUNCT":
                res["tag_" + token.tag_ + "_num"] = res.get("tag_" + token.tag_ + "_num", 0) + 1

        return res

    #Расчет количества всех видов связей, без пунктуации
    @staticmethod
    def _get_dependencies_count(sentence):
        res = dict()
        for token in sentence:
            if token.dep_ != "_SP" and token.dep_ != "root" and token.dep_ != "punct":
                res["dep_" + token.dep_ + "_num"] = res.get("dep_" + token.dep_ + "_num", 0) + 1

        return res

    