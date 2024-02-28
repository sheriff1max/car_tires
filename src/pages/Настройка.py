import streamlit as st
import tkinter as tk
from tkinter import filedialog

from utils import constants
from utils import load_state, save_state, read_classes_from_txt
from ml import Api

st.session_state = load_state(constants.PICKLE_FILENAME, st.session_state)

root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)

st.title('Настройка ПО')

# Загрузка модели
if st.button('Загрузка модели', type='primary'):
    path_model = st.text_input('Путь выбранной модели:', filedialog.askopenfilename(master=root))
    st.session_state['path_model'] = path_model

    classes = read_classes_from_txt(constants.CLASSES_TXT)
    st.session_state['classes'] = classes

    api = Api(path_model, classes)
    st.session_state['api'] = api

    save_state(constants.PICKLE_FILENAME, st.session_state)
else:
    path_model = '' if 'path_model' not in st.session_state else st.session_state['path_model']
    path_model = st.text_input('Путь выбранной модели:', path_model)

# Кол-во выводимых результатов на сайте.
max_count_show_img_ui = st.number_input('Максимальное количество выводимых результатов', min_value=1, value=constants.MAX_COUNT_SHOW_IMG_UI)
if max_count_show_img_ui != st.session_state.get('max_count_show_img_ui', None):
    st.session_state['max_count_show_img_ui'] = max_count_show_img_ui
    save_state(constants.PICKLE_FILENAME, st.session_state)

if 'api' in st.session_state:
    st.subheader('ПО готово к работе!')
    st.text('Можете переходить в другую вкладку для работы.')
