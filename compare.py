import argparse
import pickle
import ast
import warnings
import inspect # нужно, чтобы получить методы библиотеки ast
import numpy as np
import pandas as pd
import catboost as cb
from sklearn.feature_extraction.text import TfidfVectorizer

class Compare():
    # Класс предсказания уровня похожести двух кодов

    def __init__(self, input_path):
        # Инициализирует поля с массивами кодов
        self.files = []
        self.plagiat1 = []
        self.plagiat2 = []

        with open(input_path, 'r', encoding='utf-8') as file:
            files = file.readlines()

        files = [file.strip().split(' ') for file in files]
        files_path = [file[0] for file in files]
        plagiat1_path = [file[1] for file in files]

        for i in files_path:
            with open(i, 'r', encoding='utf-8') as f:
                code = f.read()
                self.files.append(code)

        for i in (plagiat1_path):
            with open(i, 'r', encoding='utf-8') as f:
                code = f.read()
                self.plagiat1.append(code)

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

    def code_analysis(self, flag=1):
        ''' Проводим анализ текстов кодов.
        Возвращает кортеж с двумя массивами косинусных расстояний между
        полученными преобразованиями кодов'''

        dict1 = []
        dict2 = []
        files_indexes = []
        plag1_indexes = []
        plag2_indexes = []
        range_len = 1
        for q in range(range_len):
            
            rng = np.random.RandomState(q)
            for i in range(0, len(self.files)):

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

    def names(self, flag=1, files_indexes=None, plag1_indexes=None,
              plag2_indexes=None):
        '''Проводим анализ имен, используемых в кодах.
        Возвращает кортеж с двумя массивами косинусных расстояний между
        полученными преобразованиями кодов'''

        dict1 = []
        dict2 = []
        range_len = len(self.files)

        for i in range(0, range_len):

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
        '''Проводим анализ структуры кодов.
        Возвращает кортеж с двумя массивами косинусных расстояний между
        полученными преобразованиями кодов'''

        dict1 = []
        dict2 = []

        range_len = len(self.files)

        for i in range(0, range_len):

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

    def to_dataset(self):
        # Создает датасет из входных данных
        data_predict = pd.DataFrame()
        vert = self.get_vert()

        for i in range(len(self.files)):
            file_cur = self.drop_comment(self.files[i])
            plag1_cur = self.drop_comment(self.plagiat1[i])

            df_file = self.make_df(file_cur, [i], sub='_file')
            df_plag1 = self.make_df(plag1_cur, [i], sub='_plagiat')

            df1 = pd.merge(df_file, df_plag1, left_index=True, right_index=True)
            data_predict = pd.concat([data_predict, df1])

        feature_code_analysis = self.code_analysis(1)
        feature_names = self.names(1)
        feature_structure = self.structure(1, vert)

        ast_features1 = pd.DataFrame({'code_analysis': feature_code_analysis[0],
                                      'names': feature_names[0],
                                      'structure': feature_structure[0]},
                                     index=np.arange(len(self.files)))

        data_predict = pd.merge(data_predict, ast_features1, left_index=True,
                                right_index=True)
        data_predict = data_predict.fillna(-1)
        # Избавляемся от недопустимых значений в столбцах
        data_predict.loc[
            data_predict[data_predict['code_analysis'] == -1].index, 'code_analysis'] = \
            data_predict.loc[data_predict[data_predict['code_analysis'] == -1].index, 'names']
        data_predict.loc[
            data_predict[data_predict['code_analysis'] == -1].index, 'code_analysis'] = \
            data_predict.loc[data_predict[data_predict['code_analysis'] == -1].index, 'structure']
        data_predict.loc[
            data_predict[data_predict['names'] == -1].index, 'names'] = \
            data_predict.loc[data_predict[data_predict['names'] == -1].index, 'code_analysis']
        data_predict.loc[
            data_predict[data_predict['structure'] == -1].index, 'structure'] = \
            data_predict.loc[data_predict[data_predict['structure'] == -1].index, 'code_analysis']

        return data_predict

    def predict_model(self, pkl_filename, answer_filename):
        # Предсказывает уровень похожести двух кодов
        # и записывает ответ в файл
        with open(pkl_filename, 'rb') as file:
            model = pickle.load(file)

        data = self.to_dataset()
        predict = model.predict(data)
        data_sum = data[['code_analysis', 'names', 'structure']].sum(axis=1)
        answer = (data_sum + predict) / 4

        with open(answer_filename, 'w') as f:
            for ans in answer.values:
                f.write(str(ans.round(3)) + '\n')

        return True

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('input', type=str, help='Input file with codes')
parser.add_argument('scores', type=str, help='Output file with scores')
parser.add_argument('--model', type=str, help='Model filename')
args = parser.parse_args()

input_path = args.input
model = Compare(input_path)
answer_path = args.scores
pkl_filename = args.model
ans = model.predict_model(pkl_filename, answer_path)
