import os
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# ---------- โหลด API KEY ----------
load_dotenv()
API_KEY = os.getenv("gemini_api_key")

if not API_KEY:
    raise ValueError("ไม่พบ GOOGLE_API_KEY ใน .env")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ---------- โหลดไฟล์ ----------
df_task = pd.read_csv("suvarnabhumi_expansion_schedule.csv")
df_worker = pd.read_csv("worker_data.csv")

# เอาเฉพาะ skill ที่ไม่ว่างและไม่ซ้ำ
df_worker_skill = df_worker['skill_type_name'].dropna().unique().tolist()

# สร้างรายการงานเป็นข้อความแบบอ่านง่าย
task_list = []
for _, row in df_task.iterrows():
    task_name = str(row['task_name']).strip()
    sub_task = str(row['sub_task']).strip()
    task_list.append(f"- งาน: {task_name}, รายละเอียดงานย่อย: {sub_task}")

# รวมทั้งหมดเป็น prompt
prompt = f"""
คุณเป็นผู้เชี่ยวชาญด้านการจับคู่ทักษะของแรงงานกับงานที่ต้องทำ
โดยมีทักษะของแรงงานดังต่อไปนี้:

{', '.join(df_worker_skill)}

และรายการงานที่ต้องการคนทำดังนี้:
{chr(10).join(task_list)}

กรุณาจับคู่ให้ว่าแต่ละงานควรใช้ทักษะใดจากรายการที่มีมากที่สุด
ถ้าไม่สามารถจับคู่ได้ ให้ตอบว่า "ไม่สามารถจับคู่ได้"

แสดงผลในรูปแบบตารางที่มี 3 คอลัมน์: 
- ชื่องาน
- งานย่อย
- ทักษะที่จับคู่ได้
"""

# ส่ง prompt ไปยัง LLM
try:
    response = model.generate_content(prompt)
except Exception as e:
    print(f"❌ ข้อผิดพลาดในการเรียก LLM: {e}")
    exit(1)

# DEBUG: ตรวจสอบ response
print("🧾 ตอบกลับจาก LLM:\n", response.text)

# ------------ แปลงผลลัพธ์เป็น DataFrame ------------
results = []
lines = response.text.split('\n')
header_found = False
for line in lines:
    print(f"LINE: {line}")
    # ข้ามบรรทัดว่างหรือ header
    if line.strip() and '|' in line and not line.startswith('|-'):
        if not header_found and 'ชื่องาน' in line:
            header_found = True
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 3:
            task_name, sub_task, skill = parts[:3]
            # ค้นหา start_date และ end_date จาก df_task
            match_row = df_task[
                (df_task['task_name'].astype(str).str.strip() == task_name) &
                (df_task['sub_task'].astype(str).str.strip() == sub_task)
            ]
            if not match_row.empty:
                start_date = match_row.iloc[0]['start_date']
                end_date = match_row.iloc[0]['end_date']
            else:
                print(f"⚠️ ไม่พบการจับคู่สำหรับ งาน: {task_name}, งานย่อย: {sub_task}")
                start_date = "ไม่พบ"
                end_date = "ไม่พบ"
            results.append({
                "task_name": task_name,
                "sub_task": sub_task,
                "matched_skill": skill,
                "start_date": start_date,
                "end_date": end_date
            })

# สร้าง DataFrame และบันทึก
matched_df = pd.DataFrame(results)
if not matched_df.empty:
    matched_df.to_csv("matched_skills.csv", index=False, encoding='utf-8')
    print("✅ บันทึก matched_skills.csv เรียบร้อยแล้ว")
    print("\nตัวอย่างข้อมูลใน matched_skills.csv:")
    print(matched_df.head())
else:
    print("⚠️ ไม่มีข้อมูลจะบันทึก")