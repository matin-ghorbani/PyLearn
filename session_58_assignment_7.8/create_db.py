import sqlite3 as sl
import argparse
# from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--db-name', type=str, default='license_plate_db.db', help='Path of your database')
opt = parser.parse_args()

names: list[str] = [
    'Celestino Schuster',
    'Micheal Schumm',
    'Greg Jones'
]

license_plates: list[str] = [
    '13n73199',
    '12p73289',
    '98i37829'
]

conn = sl.connect(opt.db_name)
curr = conn.cursor()
create_table_sql = "CREATE TABLE IF NOT EXISTS masters(id INTEGER PRIMARY KEY, name TEXT, plate TEXT)"
curr.execute(create_table_sql)


add_master_sql = "insert into masters (name, plate) values (?, ?)"
for name, plate in zip(names, license_plates):
    curr.execute(add_master_sql, (name, plate))
    print(f'Master {name} => {plate} Added.')

conn.commit()
print('All Masters Added')
conn.close()
