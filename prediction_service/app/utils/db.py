import sqlite3
import os

DB_PATH = "/app/database/users.db"

def init_db():
    if not os.path.exists("/app/database"):
        os.makedirs("/app/database")
    conn = sqlite3.connect(DB_PATH)
    # Create table if not exists
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    );
    """)
    # Insert a test user if not exists
    conn.execute("INSERT OR IGNORE INTO users (username,password) VALUES (?,?)", ("user1","xxxxxx"))
    conn.commit()
    conn.close()

def authenticate_user(username: str, password: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    if row and row[0] == password:
        return True
    return False
