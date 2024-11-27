import mysql.connector

# Conectare la baza de date MySQL
db = mysql.connector.connect(
    host="localhost",
    user="aaa",
    password="1234",
    database="TEST_BD"
)

cursor = db.cursor()

# Execută o interogare
cursor.execute("SELECT * FROM pers")

# Afișează rezultatele
for row in cursor.fetchall():
    print(row)

# Închide conexiunea
cursor.close()
db.close()
