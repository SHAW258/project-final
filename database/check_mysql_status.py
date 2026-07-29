import pymysql

def safe_print(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='ignore').decode('ascii'), flush=True)

safe_print("=== Checking MySQL Status ===")
try:
    conn = pymysql.connect(host='localhost', port=3306, user='admin', password='Indrajit67@', connect_timeout=3)
    safe_print("✅ MySQL Server Connection: ACTIVE & SUCCESSFUL!")
    safe_print("   Host: localhost:3306")
    safe_print("   User: admin")

    with conn.cursor() as cursor:
        cursor.execute("SHOW DATABASES;")
        dbs = cursor.fetchall()
        safe_print(f"   Databases found: {[d[0] for d in dbs]}")
    conn.close()

except Exception as e:
    safe_print(f"❌ Connection error: {e}")
