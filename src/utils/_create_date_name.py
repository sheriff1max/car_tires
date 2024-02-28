import datetime


def create_date_name(name: str, extension: str = None) -> str:
    """"""
    now_datetime = []
    for symbol in str(datetime.datetime.now()):
        now_datetime.append(
            symbol if symbol not in (" ", ".", ":") else "-"
        )
    name = f"{name}_{''.join(now_datetime)}"

    if extension:
        if '.' in extension:
            name += extension
        else:
            name = f'{name}.{extension}'
    return name
