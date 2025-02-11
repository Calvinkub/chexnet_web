import sqlite3
import os

def create_table():
    # ระบุเส้นทางที่ชัดเจนเพื่อให้แน่ใจว่าฐานข้อมูลถูกสร้างในโฟลเดอร์ที่ต้องการ
    db_path = os.path.join(os.path.dirname(__file__), 'Image_data.db')
    image_database = sqlite3.connect(db_path)
    data = image_database.cursor()
    data.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image BLOB
        )
    """)
    image_database.commit()
    image_database.close()

if __name__ == "__main__":
    create_table()
    print("Table created successfully.")
    print(f"Database location: {os.path.join(os.path.dirname(__file__), 'Image_data.db')}")


