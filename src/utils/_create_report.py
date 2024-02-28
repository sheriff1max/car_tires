import pandas as pd
from ._create_date_name import create_date_name


def create_report(
    filename: str,
    column_data: list[list],
    column_names: list[str],
) -> str:
    if '.' in filename:
        filename = filename[:filename.index('.')]

    generated_filename = create_date_name(filename, '.csv')

    data = {column_names[i]: column_data[i] for i in range(len(column_data))}
    df = pd.DataFrame(data)
    df.to_csv(generated_filename, index=False, encoding='utf-8-sig')
    return generated_filename
