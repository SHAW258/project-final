import pymysql

def safe_print(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='ignore').decode('ascii'), flush=True)

safe_print("=== Testing MySQL connection for user 'root' ===")
try:
    conn = pymysql.connect(host='localhost', port=3306, user='root', password='indrajit', connect_timeout=3)
    safe_print("✅ Connected to MySQL Server at localhost:3306 with user='root'!")
    with conn.cursor() as cursor:
        cursor.execute("CREATE DATABASE IF NOT EXISTS `aqi_db` DEFAULT CHARACTER SET utf8mb4;")
        cursor.execute("SHOW DATABASES;")
        dbs = cursor.fetchall()
        safe_print(f"   Databases found in MySQL: {[d[0] for d in dbs]}")
    conn.commit()
    conn.close()
    safe_print("✅ Database `aqi_db` created/verified successfully in MySQL!")
except Exception as e:
    safe_print(f"❌ Error: {e}")
