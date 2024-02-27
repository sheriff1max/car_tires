import streamlit as st
import tkinter as tk
from tkinter import filedialog
import os

from utils import constants
from utils import load_state
from ml.utils import load_file
from ml import Api
from ml.interpret import interpret_integrated_gradients, interpret_grad_cam


def get_api() -> Api:
    return st.session_state['api']


st.session_state = load_state(constants.PICKLE_FILENAME, st.session_state)
if 'api' not in st.session_state:
    st.subheader('Сперва нужно настроить ПО!')
    st.text('Зайдите, пожалуйста, во вкладку `Настройки`')

else:
    api = get_api()

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    st.title('Проверка шин по фото')

    path = None
    flag_file, flag_folder = False, False
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Загрузка одной фотографии'):
            path = st.text_input('Путь до фото:', filedialog.askopenfilename(master=root))
            flag_file = True

    with col2:
        if st.button('Загрузка папки с фотографиями'):
            path = st.text_input('Путь до папки:', filedialog.askdirectory(master=root))
            flag_folder = True

    interpret_algorithm = st.radio(
        'Алгоритм для интерпретация результатов:',
        ['Отсутвует', 'Integrated Gradients', 'GradCAM']
    )

    if path and flag_file:
        paths = [path]
        list_results = api.predict(paths)

    elif path and flag_folder:
        paths = [os.path.join(path, filename) for filename in os.listdir(path)]
        list_results = api.predict(paths)

    if path and (flag_file or flag_folder):
        # TODO: https://docs.streamlit.io/knowledge-base/using-streamlit/how-download-file-streamlit
        for i, result in enumerate(list_results, start=1):

            if interpret_algorithm == 'Integrated Gradients':
                image = interpret_integrated_gradients(
                    api._model,
                    load_file(result.get_path()),
                    result.get_image(),
                    result.get_label_idx(),
                )
                st.image(image, clamp=True)
            elif interpret_algorithm == 'GradCAM':
                image = interpret_grad_cam(
                    api._model,
                    load_file(result.get_path()),
                    result.get_image(),
                    result.get_label_idx(),
                )
                st.image(image, clamp=True)
            else:
                st.image(result.get_image(), clamp=True, channels='RGB')

            st.text(f'Файл: {result.get_path()}')
            st.markdown(f'Предсказанный класс: **{result.get_label()}**')
            st.markdown(f'Достоверность предсказания: **{result.get_prob()}%**')

            if i >= constants.MAX_COUNT_SHOW_IMG_UI:
                st.subheader('Количество выводимых результатов ограничено.')
                st.text('Больше результатов смотри в файле.')
                break
