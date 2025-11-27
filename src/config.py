import os

class Config:
    DATA_DIR = "data"           # پوشه دیتاست
    FD = "FD001"                # نوع دیتاست
    RUL_THRESHOLD = 30          # حد آستانه برچسب خطر
    VAL_SIZE = 0.2              # درصد ولیدیشن
    RANDOM_STATE = 42           # Seed
    OUTPUT_DIR = "outputs"      # پوشه خروجی
    TRUST_SAMPLES = 50          # تعداد نمونه برای CSV
