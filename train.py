import argparse
from os import listdir
import pickle
import warnings
import ast
import inspect # нужно, чтобы получить методы библиотеки ast
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

class Train():
    # Класс тренировки модели для вычисления уровня похожести двух кодов

    def __init__(self, files_path, plagiat1_path, plagiat2_path):
        # Инициализирует поля с массивами кодов оригиналов и плагиатов
        self.files = []
        self.plagiat1 = []
        self.plagiat2 = []

        for i in sorted(listdir(files_path)):
            with open(files_path + i, encoding='utf-8') as f:
                code = f.read()
                self.files.append(code)

        for i in sorted(listdir(plagiat1_path)):
            with open(plagiat1_path + i, encoding='utf-8') as f:
                code = f.read()
                self.plagiat1.append(code)

        for i in sorted(listdir(plagiat2_path)):
            with open(plagiat2_path + i, encoding='utf-8') as f:
                code = f.read()
                self.plagiat2.append(code)

    def drop_comment(self, elem):
        # Удаляет комментарии из кода
        # Т.к. AST удается построить не всегда, то также будем использовать этот метод
        elem = elem.split('"""')
        elem = [elem[i] for i in range(0, len(elem), 2)]
        elem = ' '.join(elem)
        elem = elem.split('\n')
        elem = [i for i in elem if (i.strip() and i.lstrip()[0] != '#')]
        elem = '\n'.join(elem)
        return elem

    def replace_syntax(self, string):
        # Удаляет лишние символы из строки
        syntax = ":().,=+*&|\[]'"
        for i in syntax + '"':
            string = string.replace(i, ' ', -1)

        return string

    def tokenize_text(self, string):
        # Разбивает текст кода на токены
        symbols = ['\n', '\t']
        string = string.replace('\n', ' ', -1)
        string = string.replace('\t', ' ', -1)
        string = string.replace('"', "'", -1)
        tokens = string.split(' ')
        tokens = [token for token in tokens if (token)]

        return tokens

    def code_analysis(self, flag):
        ''' Если flag=1, то проводим анализ кодов, являющихся плагиатами,
        если flag=0, то проводим анализ кодов, не являющихся плагиатами (выбор
        пар неплагиатных кодов производим рандомно)

        Возвращает кортеж с двумя массивами косинусных расстояний между
        полученными преобразованиями кодов'''

        dict1 = []
        dict2 = []
        files_indexes = []
        plag1_indexes = []
        plag2_indexes = []

        # если flag=0, то будем 10 раз прогонять код, чтобы взять много пар
        # неплагиатных кодов и затем из них выбрать наиболее похожие
        range_len = 1
        if (not flag):
            range_len = 10

        for q in range(range_len):

            rng = np.random.RandomState(q)
            for i in range(0, len(self.files)):

                if (not flag):
                    a, b = i, i + rng.randint(0 - i + 1,len(self.files) - i - 1)
                    c = i + rng.randint(0 - i + 1, len(self.files) - i - 1)
                    if (a == b):
                        b += 1
                    if (a == c):
                        c += 1

                    files_indexes.append(a)
                    plag1_indexes.append(b)
                    plag2_indexes.append(c)
                else:
                    a, b, c = i, i, i

                try:
                    # Для файлов plagiat1 и files
                    sent = self.replace_syntax(self.drop_comment(self.files[a]))
                    sent1 = self.replace_syntax(
                        self.drop_comment(self.plagiat1[b]))
                    sent = [sent, sent1]
                    vectorizer = TfidfVectorizer(
                        tokenizer=lambda x: self.tokenize_text(x),
                        max_features=5000,
                        ngram_range=(1, 3))
                    body = vectorizer.fit_transform(sent)
                    df = pd.DataFrame(body.toarray(),
                                      columns=vectorizer.get_feature_names_out())

                    maxx = max(len(self.drop_comment(self.files[a])),
                               len(self.drop_comment(self.plagiat1[b])))
                    minn = min(len(self.drop_comment(self.files[a])),
                               len(self.drop_comment(self.plagiat1[b])))
                    diff = minn / maxx

                    # Косинусное расстояние с учетом длин файлов
                    cos1 = sum(df.iloc[0].values.reshape(-1, 1) * \
                               df.iloc[1].values.reshape(-1, 1)) * diff / \
                               ((sum(df.iloc[0].values.reshape(-1, 1) ** 2) ** 0.5) * \
                               (sum(df.iloc[1].values.reshape(-1, 1) ** 2) ** 0.5))

                    dict1.append(cos1[0])
                except:
                    dict1.append(-1)

                try:
                    # Для файлов plagiat2 и files
                    sent1 = self.replace_syntax(
                        self.drop_comment(self.plagiat2[c]))
                    sent = self.replace_syntax(self.drop_comment(self.files[a]))
                    sent = [sent, sent1]
                    vectorizer = TfidfVectorizer(
                        tokenizer=lambda x: self.tokenize_text(x),
                        max_features=5000,
                        ngram_range=(1, 3))
                    body = vectorizer.fit_transform(sent)
                    df = pd.DataFrame(body.toarray(),
                                      columns=vectorizer.get_feature_names_out())

                    maxx = max(len(self.drop_comment(self.files[a])),
                               len(self.drop_comment(self.plagiat2[c])))
                    minn = min(len(self.drop_comment(self.files[a])),
                               len(self.drop_comment(self.plagiat2[c])))
                    diff = minn / maxx

                    # Косинусное расстояние с учетом длин файлов
                    cos2 = sum(df.iloc[0].values.reshape(-1, 1) * \
                               df.iloc[1].values.reshape(-1, 1)) * diff / \
                               ((sum(df.iloc[0].values.reshape(-1, 1) ** 2) ** 0.5) * \
                               (sum(df.iloc[1].values.reshape(-1, 1) ** 2) ** 0.5))

                    dict2.append(cos2[0])
                except:
                    dict2.append(-1)

        if (flag):
            return dict1, dict2
        else:
            return {'files_indexes': np.array(files_indexes),
                    'plag1_indexes': np.array(plag1_indexes),
                    'plag2_indexes': np.array(plag2_indexes),
                    'dict1': np.array(dict1),
                    'dict2': np.array(dict2),
                    }

    # ------------------------------------------------ Code names with AST

    def clear(self, string):
        # очищает код, строя Abstract syntax tree
        string = self.drop_comment(string)
        string = string.replace("'t ", ' ', -1)  # чтобы избежать ошибки при исп. doesn't
        parse = ast.parse(string)
        unparse = ast.unparse(parse)
        unparse = unparse.replace('"', "'", -1)

        return ast.dump(ast.parse(unparse))

    def replace_syntax_new(self, string):
        # удаляет личшние символы
        syntax = ":()=.,+*&|\[]"
        for i in syntax:
            string = string.replace(i, ' ', -1)

        string = string.split(' ')
        string = [s.strip() for s in string if s.strip()]

        return ' '.join(string)

    def tokenize_names(self, string):
        # токенизирует все имена, встречающиеся в коде
        # (имена методов, функций, переменных и тд)
        names_list = self.replace_syntax_new(self.clear(string))
        names_list = names_list.split("'")
        names_list = [names_list[i] for i in range(1, len(names_list), 2)]
        names_string = ' '.join(names_list)

        syntax = ":()=.,+*&|\[]_"
        for i in syntax:
            names_string = names_string.replace(i, '', -1)

        names_string = names_string.lower()
        names_list = names_string.split(' ')
        names_list = np.array(names_list)

        return names_list

    def names(self, flag, files_indexes=None, plag1_indexes=None,
              plag2_indexes=None):
        ''' Если flag=1, то проводим анализ имен, используемых в кодах,
        являющихся плагиатами, если flag=0, то проводим анализ имен,
        используемых в кодах, не являющихся плагиатами (индексы пар неплагиатных
        кодов берем по порядку из массивов file_indexes, plag1_indexes и
        plag2_indexes)

        Возвращает кортеж с двумя массивами косинусных расстояний между
        полученными преобразованиями кодов'''

        dict1 = []
        dict2 = []

        range_len = len(self.files)
        if (not flag):
            range_len = files_indexes.shape[0]

        for i in range(0, range_len):
            if (not flag):
                a, b, c = files_indexes[i], plag1_indexes[i], plag2_indexes[i]
            else:
                a, b, c = i, i, i

            try:
                # Для файлов plagiat1 и files
                sent = self.tokenize_names(self.files[a])
                sent1 = self.tokenize_names(self.plagiat1[b])
                maxx = max(len(sent1), len(sent))
                minn = min(len(sent1), len(sent))
                sent = [self.files[a], self.plagiat1[b]]
                vectorizer = TfidfVectorizer(
                    tokenizer=lambda x: self.tokenize_names(x),
                    max_features=5000,
                    ngram_range=(1, 1))
                body = vectorizer.fit_transform(sent)
                df = pd.DataFrame(body.toarray(),
                                  columns=vectorizer.get_feature_names_out())

                diff = minn / maxx
                cos1 = sum(df.iloc[0].values.reshape(-1, 1) * \
                           df.iloc[1].values.reshape(-1, 1)) * diff / \
                           ((sum(df.iloc[0].values.reshape(-1, 1) ** 2) ** 0.5) * \
                           (sum(df.iloc[1].values.reshape(-1, 1) ** 2) ** 0.5))

                dict1.append(cos1[0])
            except:
                # При невозможности построения AST, будем добавлять -1
                dict1.append(-1)

            try:
                # Для файлов plagiat2 и files
                sent1 = self.tokenize_names(self.plagiat2[c])
                sent = self.tokenize_names(self.files[a])
                maxx = max(len(sent1), len(sent))
                minn = min(len(sent1), len(sent))
                sent = [self.files[a], self.plagiat2[c]]
                vectorizer = TfidfVectorizer(
                    tokenizer=lambda x: self.tokenize_names(x),
                    max_features=5000,
                    ngram_range=(1, 1))
                body = vectorizer.fit_transform(sent)
                df = pd.DataFrame(body.toarray(),
                                  columns=vectorizer.get_feature_names_out())

                diff = minn / maxx
                cos2 = sum(df.iloc[0].values.reshape(-1, 1) * \
                           df.iloc[1].values.reshape(-1, 1)) * diff / \
                           ((sum(df.iloc[0].values.reshape(-1, 1) ** 2) ** 0.5) * \
                           (sum(df.iloc[1].values.reshape(-1, 1) ** 2) ** 0.5))

                dict2.append(cos2[0])
            except:
                # При невозможности построения AST, будем добавлять -1
                dict2.append(-1)

        return dict1, dict2

    # -------------------------------------- Code structure with AST

    def get_vert(self):
        # Получает все возможные "вершины" дерева кода из библиотеки ast
        # vert - список возможных узлов дерева кода (FOR, IF...)
        vertexes = inspect.getmembers(ast)
        vert = []
        for i in vertexes:
            vert.append(i[0])
            if i[0] == 'YieldFrom':
                break
        vert.remove('Load')
        vert = vert[1:]
        vert.append('(')
        vert.append(')')

        return vert

    def structure_syntax(self, string):
        # Удаляет и преобразует символы строки
        syntax = ":=.,+*&|\[]"
        for i in syntax:
            string = string.replace(i, ' ', -1)

        string = string.replace(')', ' ) ', -1)
        string = string.replace('(', ' ( ', -1)
        string = string.split(' ')
        string = [s.strip() for s in string if s.strip()]

        return ' '.join(string)

    def get_structure(self, string, vert):
        # Возвращает строку со структурой кода
        string = string.replace(' in: ', ' input: ', -1)
        structure_blocks = self.structure_syntax(self.clear(string))
        structure_blocks = structure_blocks.split(' ')
        structure_blocks_list = []

        for i in structure_blocks:
            if i in vert:
                structure_blocks_list.append(i)

        structure_blocks = ' '.join(structure_blocks_list)
        while ('( )' in structure_blocks
               or '()' in structure_blocks
               or '  ' in structure_blocks):
            structure_blocks = structure_blocks.replace('( )', '', -1)
            structure_blocks = structure_blocks.replace('  ', ' ', -1)
            structure_blocks = structure_blocks.replace('()', '', -1)

        return structure_blocks

    def structure(self, flag, vert, files_indexes=None, plag1_indexes=None,
                  plag2_indexes=None):
        ''' Если flag=1, то проводим анализ структуры кодов, являющихся плагиатами,
        если flag=0, то анализируем структуры кодов, не являющихся плагиатами
        (индексы пар неплагиатных кодов берем по порядку из массивов
        file_indexes, plag1_indexes и plag2_indexes)

        Возвращает кортеж с двумя массивами косинусных расстояний между
        полученными преобразованиями кодов'''

        dict1 = []
        dict2 = []

        range_len = len(self.files)
        if (not flag):
            range_len = files_indexes.shape[0]

        for i in range(0, range_len):
            if (not flag):
                a, b, c = files_indexes[i], plag1_indexes[i], plag2_indexes[i]
            else:
                a, b, c = i, i, i

            try:
                # Для файлов plagiat1 и files
                sent = self.get_structure(self.files[a], vert)
                sent1 = self.get_structure(self.plagiat1[b], vert)
                maxx = max(len(sent1), len(sent))
                minn = min(len(sent1), len(sent))
                sent = [sent, sent1]
                vectorizer = TfidfVectorizer(
                    tokenizer=lambda x: self.tokenize_text(x),
                    max_features=5000,
                    ngram_range=(7, 10))
                body = vectorizer.fit_transform(sent)
                df = pd.DataFrame(body.toarray(),
                                  columns=vectorizer.get_feature_names_out())

                diff = minn / maxx
                cos1 = sum(df.iloc[0].values.reshape(-1, 1) * \
                           df.iloc[1].values.reshape(-1, 1)) * diff / \
                           ((sum(df.iloc[0].values.reshape(-1, 1) ** 2) ** 0.5) * \
                           (sum(df.iloc[1].values.reshape(-1, 1) ** 2) ** 0.5))

                dict1.append(cos1[0])
            except:
                # При невозможности построения AST, будем добавлять -1
                dict1.append(-1)

            try:
                # Для файлов plagiat2 и files
                sent1 = self.get_structure(self.plagiat2[c], vert)
                sent = self.get_structure(self.files[a], vert)
                maxx = max(len(sent1), len(sent))
                minn = min(len(sent1), len(sent))
                sent = [sent, sent1]
                vectorizer = TfidfVectorizer(
                    tokenizer=lambda x: self.tokenize_text(x),
                    max_features=5000,
                    ngram_range=(7, 10))
                body = vectorizer.fit_transform(sent)
                df = pd.DataFrame(body.toarray(),
                                  columns=vectorizer.get_feature_names_out())

                diff = minn / maxx
                cos2 = sum(df.iloc[0].values.reshape(-1, 1) * \
                           df.iloc[1].values.reshape(-1, 1)) * diff / \
                           ((sum(df.iloc[0].values.reshape(-1, 1) ** 2) ** 0.5) * \
                           (sum(df.iloc[1].values.reshape(-1, 1) ** 2) ** 0.5))
                dict2.append(cos2[0])
            except:
                # При невозможности построения AST, будем добавлять -1
                dict2.append(-1)

        return dict1, dict2

    # ------------------------------------------- Other features

    def functions(self, string):
        # Количество функций
        return string.count(' def ') + string.count('\ndef ')

    def classes(self, string):
        # Количество классов
        return string.count(' class ') + string.count('\nclass ')

    def modules(self, string):
        # Количество модулей
        return string.count('import ')

    def metods(self, string):
        # Количество обращений к методам
        return string.count('.')

    def conditions(self, string):
        # Количество условий
        return string.count(' if ') + string.count('\nif ')

    def whitespaces(self, string):
        # Количество пробелов
        return string.count(' ')

    def indents(self, string):
        # Количество переходов на новую строку
        return string.count('\n')

    def make_df(self, string, index, sub):
        # Создает DataFrame с признаками, указанными выше
        function_cur = self.functions(string)
        class_cur = self.classes(string)
        module_cur = self.modules(string)
        metod_cur = self.metods(string)
        condition_cur = self.conditions(string)
        whitespace_cur = self.whitespaces(string)
        indent_cur = self.indents(string)

        df = pd.DataFrame({'functions' + sub: function_cur,
                           'classes' + sub: class_cur,
                           'modules' + sub: module_cur,
                           'metods' + sub: metod_cur,
                           'conditions' + sub: condition_cur,
                           'whitespaces' + sub: whitespace_cur,
                           'indents' + sub: indent_cur}, index=index)

        return df

    def plagiat_dataset(self):
        # Создает датасет с "плагиатными" выборками

        data1 = pd.DataFrame()
        data2 = pd.DataFrame()
        vert = self.get_vert()

        for i in range(len(self.files)):
            file_cur = self.drop_comment(self.files[i])
            plag1_cur = self.drop_comment(self.plagiat1[i])
            plag2_cur = self.drop_comment(self.plagiat2[i])

            df_file = self.make_df(file_cur, [i], sub='_file')
            df_plag1 = self.make_df(plag1_cur, [i], sub='_plagiat')
            df_plag2 = self.make_df(plag2_cur, [i], sub='_plagiat')

            df1 = pd.merge(df_file, df_plag1, left_index=True, right_index=True)
            df2 = pd.merge(df_file, df_plag2, left_index=True, right_index=True)

            data1 = pd.concat([data1, df1])
            data2 = pd.concat([data2, df2])

        feature_code_analysis = self.code_analysis(1)
        feature_names = self.names(1)
        feature_structure = self.structure(1, vert)

        ast_features1 = pd.DataFrame({'code_analysis': feature_code_analysis[0],
                                      'names': feature_names[0],
                                      'structure': feature_structure[0]},
                                     index=np.arange(len(self.files)))

        ast_features2 = pd.DataFrame({'code_analysis': feature_code_analysis[1],
                                      'names': feature_names[1],
                                      'structure': feature_structure[1]},
                                     index=np.arange(len(self.files)))

        data1 = pd.merge(data1, ast_features1, left_index=True,
                         right_index=True)
        data2 = pd.merge(data2, ast_features2, left_index=True,
                         right_index=True)

        data = pd.concat([data1, data2])
        data['is_plagiat'] = 1  # 1 - выборка является плагиатной

        return data

    def original_dataset(self):
        # Создает датасет с "неплагиатными" выборками

        originals = self.code_analysis(0)
        originals = pd.DataFrame(originals)
        originals = originals.dropna()
        vert = self.get_vert()

        original1 = originals[['files_indexes', 'plag1_indexes', 'dict1']]
        original2 = originals[['files_indexes', 'plag2_indexes', 'dict2']]

        original1 = original1.drop_duplicates(
            subset=['files_indexes', 'plag1_indexes'])
        original2 = original2.drop_duplicates(
            subset=['files_indexes', 'plag2_indexes'])

        # берем похожие коды
        original1_high = original1[original1['dict1'] > 0.15]
        original2_high = original2[original2['dict2'] > 0.15]

        # берем непохожие коды
        original1_low = original1[original1['dict1'] < 0.15].sample(150,
                                                                    random_state=0)
        original2_low = original2[original2['dict2'] < 0.20].sample(150,
                                                                    random_state=0)

        files_indexes_new = np.concatenate([original1_high['files_indexes'],
                                            original1_low['files_indexes'],
                                            original2_high['files_indexes'],
                                            original2_low['files_indexes']])

        plag1_indexes_new = np.concatenate([original1_high['plag1_indexes'],
                                            original1_low['plag1_indexes']])

        plag2_indexes_new = np.concatenate([original2_high['plag2_indexes'],
                                            original2_low['plag2_indexes']])

        files_indexes_new1 = files_indexes_new[:plag1_indexes_new.shape[0]]
        files_indexes_new2 = files_indexes_new[plag1_indexes_new.shape[0]:]

        feature_names1 = self.names(0, files_indexes=files_indexes_new1,
                                    plag1_indexes=plag1_indexes_new,
                                    plag2_indexes=plag1_indexes_new)[0]

        feature_structure1 = \
        self.structure(0, vert, files_indexes=files_indexes_new1,
                       plag1_indexes=plag1_indexes_new,
                       plag2_indexes=plag1_indexes_new)[0]

        feature_names2 = self.names(0, files_indexes=files_indexes_new2,
                                    plag1_indexes=plag1_indexes_new,
                                    plag2_indexes=plag1_indexes_new)[1]

        feature_structure2 = \
        self.structure(0, vert, files_indexes=files_indexes_new2,
                       plag1_indexes=plag2_indexes_new,
                       plag2_indexes=plag2_indexes_new)[1]

        data1 = pd.DataFrame()
        data2 = pd.DataFrame()

        for i in range(files_indexes_new1.shape[0]):
            file_cur = self.drop_comment(self.files[files_indexes_new1[i]])
            plag1_cur = self.drop_comment(self.plagiat1[plag1_indexes_new[i]])

            df_file = self.make_df(file_cur, [i], sub='_file')
            df_plag1 = self.make_df(plag1_cur, [i], sub='_plagiat')

            df1 = pd.merge(df_file, df_plag1, left_index=True, right_index=True)
            data1 = pd.concat([data1, df1])

        for i in range(files_indexes_new2.shape[0]):
            file_cur = self.drop_comment(self.files[files_indexes_new2[i]])
            plag2_cur = self.drop_comment(self.plagiat2[plag2_indexes_new[i]])

            df_file = self.make_df(file_cur, [i], sub='_file')
            df_plag2 = self.make_df(plag2_cur, [i], sub='_plagiat')

            df2 = pd.merge(df_file, df_plag2, left_index=True, right_index=True)
            data2 = pd.concat([data2, df2])

        original2_high.rename(columns={'plag2_indexes': 'plag1_indexes',
                                       'dict2': 'dict1'}, inplace=True)

        original2_low.rename(columns={'plag2_indexes': 'plag1_indexes',
                                      'dict2': 'dict1'}, inplace=True)

        original_new1 = pd.concat([original1_high, original1_low])
        original_new2 = pd.concat([original2_high, original2_low])

        original_new1.index = np.arange(original_new1.shape[0])
        original_new2.index = np.arange(original_new2.shape[0])

        data1 = pd.merge(original_new1, data1, left_index=True,
                         right_index=True)
        data2 = pd.merge(original_new2, data2, left_index=True,
                         right_index=True)

        data1['names'] = feature_names1
        data2['names'] = feature_names2
        data1['structure'] = feature_structure1
        data2['structure'] = feature_structure2

        data_original = pd.concat([data1, data2])
        data_original.rename(columns={'dict1': 'code_analysis'}, inplace=True)
        data_original.drop(columns=['files_indexes', 'plag1_indexes'],
                           inplace=True)
        data_original['is_plagiat'] = 0  # 1 - выборка является неплагиатной

        return data_original

    def train_model(self, pkl_filename):
        # Тренирует модель классификации и сохраняет веса
        data_plagiat = self.plagiat_dataset()
        data_original = self.original_dataset()
        data_final = pd.concat([data_plagiat, data_original])

        # Избавляемся от недопустимых значений в столбцах
        data_final.loc[
            data_final[data_final['code_analysis'] == -1].index, 'code_analysis'] = \
            data_final.loc[data_final[data_final['code_analysis'] == -1].index, 'names']
        data_final.loc[
            data_final[data_final['code_analysis'] == -1].index, 'code_analysis'] = \
            data_final.loc[data_final[data_final['code_analysis'] == -1].index, 'structure']
        data_final.loc[
            data_final[data_final['names'] == -1].index, 'names'] = \
            data_final.loc[data_final[data_final['names'] == -1].index, 'code_analysis']
        data_final.loc[
            data_final[data_final['structure'] == -1].index, 'structure'] = \
            data_final.loc[data_final[data_final['structure'] == -1].index, 'code_analysis']

        data_final_shuffled = data_final.sample(frac=1, random_state=0)

        X = data_final_shuffled.drop(columns=['is_plagiat'])
        y = data_final_shuffled['is_plagiat']
        Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.15,
                                                      stratify=y)

        # Подбор гиперпараметров осуществлялся при помощи optuna
        model = cb.CatBoostClassifier(auto_class_weights='Balanced',
                                      grow_policy='Lossguide',
                                      bootstrap_type='Bayesian',
                                      score_function='L2',
                                      bagging_temperature=1.9061212720355365,
                                      max_depth=7,
                                      n_estimators=721,
                                      learning_rate=0.04670493040512916,
                                      random_strength=1.6630686619595423,
                                      l2_leaf_reg=0.5582022002903801
                                      )

        model.fit(Xtrain, ytrain, verbose=False,
                  use_best_model=True, eval_set=(Xval, yval))

        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)

        return "The process finished successfully"

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('files', type=str, help='Input dir for original codes')
parser.add_argument('plagiat1', type=str, help='Input dir for plagiarism codes')
parser.add_argument('plagiat2', type=str, help='Input dir for plagiarism codes')
parser.add_argument('--model', type=str, help='Model filename')
args = parser.parse_args()

files_path = args.files + '/'
plagiat1_path = args.plagiat1 + '/'
plagiat2_path = args.plagiat2 + '/'
model = Train(files_path, plagiat1_path, plagiat2_path)
pkl_filename = args.model
result = model.train_model(pkl_filename)
print(result)


