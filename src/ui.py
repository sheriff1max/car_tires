import streamlit as st
import tkinter as tk
from tkinter import filedialog

import pickle
import os

from math_logic import api


st.title('Режим пользователя')

# Загрузка базы знаний:
root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)

if st.button('Загрузка базы знаний'):
    dirname = st.text_input('Выбранная база знаний:', filedialog.askopenfilename(master=root))
    filename = dirname.split('/')[-1]
    filename = os.path.join('data', filename)
    st.session_state['filename'] = filename


if 'filename' in st.session_state:
    with open(st.session_state.filename, 'rb') as f:
        data_dict = pickle.load(f)['data']

    facts = data_dict['facts']
    rules = data_dict['rules']
    operator_and = data_dict['operator_and']
    operator_or = data_dict['operator_or']
    operator_not = data_dict['operator_not']
    math_syms = [operator_and, operator_or, operator_not, '(', ')']

    st.title('Ввод коэф уверенности фактов:')
    for fact in facts.keys():
        str_fact = fact.title()
        str_fact = str_fact.replace('_', ' ')
        st.slider(f'{str_fact}?', -1.0, 1.0, 0.0, key=fact)

    if st.button('Получить результат'):
        st.title('Результат:')

        for fact in facts.keys():
            facts[fact] = st.session_state[fact]

        result_facts, facts_result_steps = api(
            facts=facts,
            rules=rules,
            operator_and=operator_and,
            operator_not=operator_not,
            math_syms=math_syms,
        )

        for i, rule in enumerate(rules):
            st.subheader(f'Вывод {i+1}')

            st.text('1. Вывод правила:')
            st.text(f'{rule} | КУ({rule.name_rule}) = {rule.confidence_coefficient_rule}')
            lst_log_list = rule.get_log_list()
            log_str = ''
            for log in lst_log_list:
                log_str += str(log)
                log_str += ' -> '
            log_str = log_str[:-4]
            st.text(log_str)

            st.text('2. Подсчёт КУ правила:')
            st.text(f'КУ({rule.result_fact_name}|{rule.name_rule}) = {rule.confidence_coefficient_formula} * {rule.confidence_coefficient_rule} = {rule.confidence_coefficient_result_fact}')

            st.text(' ')
            st.text(' ')
            st.text(' ')

        st.subheader('Вывод новых фактов из правил')

        f_x1_x2 = 'f(x1, x2) возвращает следующее:\n' \
            + '    1, если x1 = 1 или x2 = 1\n' \
            + '    -1, если x1 = -1 или x2 = -1\n' \
            + '    x1 + x2, если x1 · x2 ≤ 0\n' \
            + '    (x1 + x2) – (x1 · x2), если x1 > 0 и x2 > 0\n' \
            + '    (x1 + x2) + (x1 · x2), если x1 < 0 и x2 < 0'
        st.text(f_x1_x2)
        st.text(' ')
        st.text(' ')

        for result_fact, lst in facts_result_steps.items():
            lst_conf = lst[0]
            rule_names = ','.join(lst[1])

            text = f'КУ({result_fact}|{rule_names}) = '
            for i in range(0, len(lst_conf) - 1, 2):
                text += f'f({lst_conf[i]}, {lst_conf[i+1]}) + '

            text = text[:-3]
            text += f' = {result_facts[result_fact]}'
            st.text(text)

        st.text(' ')
        st.text(' ')
        st.subheader('КУ всех фактов:')
        st.write(result_facts)
