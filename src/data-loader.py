import sys
import pandas as pd

# تنظیم انکودینگ خروجی ترمینال به UTF-8 برای جلوگیری از خطای کاراکترهای فارسی
sys.stdout.reconfigure(encoding='utf-8')

def load_and_audit_data(url: str):
    """بارگذاری مستقیم داده‌های TLC و انجام ممیزی کیفیت اولیه"""
    print("در حال بارگذاری نمونه داده از منبع رسمی...")
    df = pd.read_parquet(url)

    print("\n--- ۱. ممیزی ساختار داده (Shape & Columns) ---")
    print(f"تعداد کل رکوردها (سطرها): {df.shape[0]:,}")
    print(f"تعداد ویژگی‌ها (ستون‌ها): {df.shape[1]}")
    print("نام ستون‌ها:", df.columns.tolist())

    print("\n--- ۲. ممیزی مقادیر گمشده (Missing Values) ---")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])

    print("\n--- ۳. ممیزی مقادیر ناممکن و پرت (Data Quality Audit) ---")
    invalid_distance = df[df['trip_distance'] < 0]
    invalid_fare = df[df['fare_amount'] < 0]
    zero_distance_high_fare = df[(df['trip_distance'] == 0) & (df['fare_amount'] > 50)]

    print(f"تعداد سفرهای با مسافت منفی: {len(invalid_distance)}")
    print(f"تعداد سفرهای با کرایه منفی: {len(invalid_fare)}")
    print(f"تعداد سفرهای با مسافت صفر و کرایه بالا (>50): {len(zero_distance_high_fare)}")
    
    return df

if __name__ == "__main__":
    # لینک نمونه داده‌های تاکسی زرد (مثلاً ژانویه ۲۰۲۴)
    sample_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
    df_sample = load_and_audit_data(sample_url)