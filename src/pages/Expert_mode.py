import streamlit as st
import tkinter as tk
from tkinter import filedialog

import pickle
import os



if 'fact_rows' not in st.session_state:
    st.session_state.fact_rows = 1
if 'rule_rows' not in st.session_state:
    st.session_state.rule_rows = 1


st.title('Режим эксперта')

# Загрузка базы знаний:
root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)

clicked = st.button('Загрузка базы знаний для редактирования')
if clicked:
    dirname = st.text_input('Выбранная база знаний:', filedialog.askopenfilename(master=root))
    filename = dirname.split('/')[-1]
    filename = os.path.join('data', filename)

    with open(filename, 'rb') as f:
        session_state_dict = pickle.load(f)['session_state']

        for key, value in session_state_dict.items():
            st.session_state[key] = value

# Настройка мат. символов:
st.title('Настройка логических символов:')

col1, col2, col3 = st.columns(3)
with col1:
    st.text_input(label="Оператор 'и'", value='&', key='and')
with col2:
    st.text_input(label="Оператор 'или'", value='|', key='or')
with col3:
    st.text_input(label="Оператор 'не'", value='-', key='not')

# Факты.
st.title('Факты для пользователя:')
col1, col2 = st.columns(2)
with col1:
    if st.button(label="Добавить факт"):
        st.session_state.fact_rows += 1
        st.experimental_rerun()
with col2:
    if st.button(label="Удалить последний факт"):
        if st.session_state.fact_rows > 1:
            st.session_state.fact_rows -= 1
            st.experimental_rerun()

for i in range(st.session_state.fact_rows):
    st.text(f'Факт {i+1}')
    st.text_input(label=" ", placeholder='Название факта', key=f'fact{i}')

# Правила.
st.title('Правила:')
st.text('Пример ввода формул (зависит от настроек логических символов):\nсильный_ветер & (вчера_был_дождь | передавали_прогноз_дождя)')
col1, col2 = st.columns(2)
with col1:
    if st.button(label="Добавить правило"):
        st.session_state.rule_rows += 1
        st.experimental_rerun()
with col2:
    if st.button(label="Удалить последнее правило"):
        if st.session_state.rule_rows > 1:
            st.session_state.rule_rows -= 1
            st.experimental_rerun()

for i in range(st.session_state.rule_rows):
    st.text(f'Правило {i+1}')
    st.text_input(label=" ", placeholder='Формула правила', key=f'rule_formula{i}')

    col1, col2 = st.columns(2)
    with col1:
        st.text_input(label=" ", placeholder='Конечный факт', key=f'rule_result_fact{i}')
    with col2:
        st.text_input(label=" ", placeholder='Коэф. уверенности', key=f'rule_confidence{i}')

# Сохранение базы знаний.
st.title('Сохранение базы знаний:')
st.text_input(label=" ", placeholder='Название файла базы знаний', key=f'save_kb')
if st.button('Сохранить'):

    if not st.session_state['save_kb']:
        st.text('Введите название файла!')
    else:
        facts = {}
        for i in range(st.session_state['fact_rows']):
            key = f'fact{i}'
            facts[st.session_state[key]] = None

        rules = []
        for i in range(st.session_state['rule_rows']):
            key_formula = f'rule_formula{i}'
            key_rule_result_fact = f'rule_result_fact{i}'
            key_rule_confidence = f'rule_confidence{i}'
            formula = parse_input_formula(st.session_state[key_formula])

            rule = Rule(
                name_rule=f'P{i}',
                confidence_coefficient_rule=float(st.session_state[key_rule_confidence]),
                formula=formula,
                result_fact_name=st.session_state[key_rule_result_fact]
            )
            rules.append(rule)

        pickle_data = {
            'session_state': {key: value for key, value in st.session_state.items()},
            'data': {
                'facts': facts,
                'rules': rules,
                'operator_and': st.session_state['and'],
                'operator_or': st.session_state['or'],
                'operator_not': st.session_state['not'],
            }
        }

        filename = st.session_state['save_kb']
        if '.pkl' not in filename:
            filename += '.pkl'
        filename = os.path.join('data', filename)

        with open(filename, 'wb') as f: 
            pickle.dump(pickle_data, f)
        st.write('Сохранено!')
