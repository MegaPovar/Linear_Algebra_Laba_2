import csv


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(title, rows):
    print(f"\n{title}")
    for row in rows:
        print(row)
