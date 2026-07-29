import os
import sys
import pymysql
import urllib.parse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load .env file from database directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=dotenv_path)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "indrajit")
DB_NAME = os.getenv("DB_NAME", "aqi_db")

def safe_print(msg):
    """Print ASCII-safe messages for Windows terminal compatibility."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='ignore').decode('ascii'))

encoded_user = urllib.parse.quote_plus(DB_USER)
encoded_password = urllib.parse.quote_plus(DB_PASSWORD)

MYSQL_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"mysql+pymysql://{encoded_user}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

Base = declarative_base()

def init_engine():
    """Attempt MySQL connection with root user; create database if missing; fallback to SQLite if needed."""
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            port=int(DB_PORT),
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=5
        )
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` DEFAULT CHARACTER SET utf8mb4;")
        connection.commit()
        connection.close()
        safe_print(f"[MySQL] Database `{DB_NAME}` created/verified successfully.")

        engine = create_engine(
            MYSQL_DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        with engine.connect() as conn:
            pass
            
        safe_print(f"[Database] Connected to MySQL database `{DB_NAME}` as `{DB_USER}` successfully!")
        return engine, True
    except Exception as e:
        safe_print(f"[Database Notice] MySQL DB Notice: {e}")
        sqlite_db_path = os.path.join(BASE_DIR, "fallback_aqi.db")
        safe_print(f"[Database] Using SQLite fallback ({sqlite_db_path}) for history storage.")
        sqlite_url = f"sqlite:///{sqlite_db_path}"
        engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
        return engine, False

engine, is_mysql = init_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """FastAPI Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
