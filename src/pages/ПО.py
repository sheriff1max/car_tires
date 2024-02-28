import streamlit as st
import tkinter as tk
from tkinter import filedialog
import os

from utils import constants, load_state, create_report
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

    st.title('Проверка качества шин по фото')

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

        column_path = []
        column_label = []
        column_prob = []

        for i, result in enumerate(list_results, start=1):
            column_path.append(result.get_path())
            column_label.append(result.get_label())
            column_prob.append(result.get_prob())

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

            if i >= st.session_state['max_count_show_img_ui']:
                st.subheader('Количество выводимых результатов ограничено.')
                st.text('Больше результатов смотри в файле.')
                break

        # Создаём отчёт.
        if not os.path.exists(constants.REPORT_FOLDER):
            os.mkdir(constants.REPORT_FOLDER)
        filename_report = create_report(
            os.path.join(constants.REPORT_FOLDER, constants.REPORT_FILENAME),
            [column_path, column_label, column_prob],
            constants.REPORT_COLUMN_NAMES,
        )

        st.text('')
        st.text('')
        st.text('')
        st.subheader(f'Полный отчёт сохранён в папке по пути `{filename_report}`')

        with open(filename_report, encoding='utf-8-sig') as f:
            file_name = filename_report.split('\\')[1]
            st.download_button('Скачать отчёт', f, type='primary', file_name=file_name)
