import os
import sqlite3
from csv import DictReader

def create_database():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link TEXT,
            price TEXT,
            TenSP TEXT,
            DG TEXT,
            SoDG TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER,
            username TEXT,
            rating INTEGER,
            comment TEXT,
            FOREIGN KEY (product_id) REFERENCES products (id)
        )
    ''')

    conn.commit()
    conn.close()

def insert_data_into_database(row_data):
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()

    # Insert product data
    cursor.execute('''
        INSERT INTO products (link, price, TenSP, DG, SoDG) VALUES (?, ?, ?, ?, ?)
    ''', (row_data['link'], row_data['price'], row_data['TenSP'], row_data['DG'], row_data['SoDG']))

    product_id = cursor.lastrowid  # Get the ID of the last inserted product

    # Insert comments data
    usernames = row_data.get('usernames', [])
    ratings = row_data.get('ratings', [])
    comments = row_data.get('comments', [])

    for i in range(len(usernames)):
        username = usernames[i].strip("[]'") if i < len(usernames) else ''
        rating = ratings[i].strip("[]'") if i < len(ratings) else ''
        comment = comments[i].strip("[]'") if i < len(comments) else ''

        cursor.execute('''
            INSERT INTO comments (product_id, username, rating, comment) VALUES (?, ?, ?, ?)
        ''', (product_id, username, rating, comment))

    conn.commit()
    conn.close()

def format_data_and_insert():
    write_headers = not os.path.exists('lazada.csv')

    with open('lazada.csv', 'r', encoding='utf-8') as file:
        csv_reader = DictReader(file)
        for row in csv_reader:
            usernames = row.get("TenHienthi", "").split(", ")  # Assume usernames are comma-separated
            ratings = row.get("comment_DG", "").split(", ")  # Assume ratings are comma-separated
            comment_contents = row.get("comment", "").split(", ")  # Assume comments are comma-separated

            # Append the data for each product to the overall data list
            row_data = {
                "link": row.get("link", ""),
                "price": row.get("price", ""),
                "TenSP": row.get("TenSP", ""),
                "DG": row.get("DG", ""),
                "SoDG": row.get("SoDG", ""),
                "usernames": usernames,
                "ratings": ratings,
                "comments": comment_contents
            }

            # Insert data into SQLite database
            insert_data_into_database(row_data)

# Create the database and tables
create_database()

# Format data from CSV and insert into SQLite database
format_data_and_insert()
