# """
# Extracts data from PostgreSQL and saves as a CSV for downstream tasks.
# """

# import pandas as pd
# import psycopg2
# import os

# def main():
#     # Load DB config from environment variables or config file
#     conn = psycopg2.connect(
#         host=os.getenv("DB_HOST", "localhost"),
#         database=os.getenv("DB_NAME", "aravind"),
#         user=os.getenv("DB_USER", "aravind"),
#         password=os.getenv("DB_PASS", "123"),
#         port=os.getenv("DB_PORT", "5432")
#     )
#     query = "SELECT * FROM salary_data"
#     df = pd.read_sql(query, conn)
#     conn.close()
#     df.to_csv("data/raw_salary_data.csv", index=False)
#     print("✅ Data extracted and saved to data/raw_salary_data.csv")

# if __name__ == "__main__":
#     main()


"""
Reads data from a CSV file and saves it as a CSV for downstream tasks.
"""

import pandas as pd
import os

def main():
    input_path = '/opt/airflow/dags/Software_Salaries.csv'  # Adjust if your file is elsewhere
    output_dir = "/opt/airflow/dags/data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "raw_salary_data.csv")
    
    df = pd.read_csv(input_path)
    df.to_csv(output_path, index=False)
    print(f"✅ Data read from {input_path} and saved to {output_path}")

if __name__ == "__main__":
    main()
