import csv


def clear_results_dir(results_dir):
    results_dir.mkdir(exist_ok=True)

    for path in results_dir.iterdir():
        if path.name == ".gitkeep":
            continue
        if path.is_file():
            path.unlink()  # удаляем старый результат перед новым запуском


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(title, rows):
    print(f"\n{title}")
    for row in rows:
        print(row)
