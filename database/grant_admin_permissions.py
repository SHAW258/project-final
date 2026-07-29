import subprocess

passwords = ['', 'root', 'Indrajit67@', '123456', 'admin', 'Indrajit', 'indrajit', 'mysql', 'root123']
mysql_exe = r"C:\Program Files\MySQL\MySQL Server 26.7\bin\mysql.exe"

for p in passwords:
    cmd = [mysql_exe, "-u", "root", f"-p{p}", "-e", "GRANT ALL PRIVILEGES ON *.* TO 'admin'@'%' WITH GRANT OPTION; FLUSH PRIVILEGES; CREATE DATABASE IF NOT EXISTS aqi_db; GRANT ALL PRIVILEGES ON aqi_db.* TO 'admin'@'%'; FLUSH PRIVILEGES;"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"✅ SUCCESS! Granted privileges using root password '{p}'")
        break
    else:
        print(f"❌ Root password '{p}' failed.")
