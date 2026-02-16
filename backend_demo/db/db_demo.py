import psycopg2
import os
from dotenv import load_dotenv

# loading vars from .env
load_dotenv()

# creating a connection from the psyco boiiiiii
conn = psycopg2.connect(
    host=os.getenv("HOST"),
    database=os.getenv("DATABASE"),
    user=os.getenv("USER"),
    password=os.getenv("PASSWORD")
)
cur = conn.cursor() # and a cursor

# Retrieve all data
# .execute gets the data
cur.execute("SELECT * FROM posts")

# .fetchone() gets one row
# .fetchall() gets all rows
# .fetchmany(n) gets n rows
data = cur.fetchall()

for row in data:
    print(row)

# should always close cursor and connection
cur.close()
conn.close()