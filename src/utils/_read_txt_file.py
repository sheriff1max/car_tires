def read_classes_from_txt(filename: str) -> list[str]:
    with open(filename, encoding='utf-8') as f:
        classes = [row.replace('\n', '') for row in f.readlines()]
    return classes
