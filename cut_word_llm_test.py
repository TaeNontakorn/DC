import pandas as pd
from collections import defaultdict

# ---------- โหลดข้อมูล ----------
df_matched = pd.read_csv("matched_skills.csv")
df_worker = pd.read_csv("worker_data.csv")

# เอาเฉพาะข้อมูลที่จำเป็น + ลบค่าว่าง
df_worker = df_worker[['worker_id','worker_name', 'skill_type_name','skill_level','evaluation_type_ratings']].dropna()

# แปลงคะแนนเป็น float สำหรับการเรียงลำดับ
df_worker['evaluation_type_ratings'] = pd.to_numeric(df_worker['evaluation_type_ratings'], errors='coerce')
df_worker = df_worker.dropna(subset=['evaluation_type_ratings'])

# ---------- สร้างโครงสร้าง map: skill → คนที่มีสกิลนั้น ----------
skill_to_workers = defaultdict(list)

for _, row in df_worker.iterrows():
    skill = row['skill_type_name'].strip()
    worker_info = {
        "worker_id": row['worker_id'],
        "worker_name": row['worker_name'],
        "skill_level": row['skill_level'],
        "evaluation_type_ratings": row['evaluation_type_ratings']
    }
    skill_to_workers[skill].append(worker_info)

# ---------- จับคู่ 1 งาน → คนที่มีทักษะตรง ----------
assigned_workers = []
top = 5  # แนะนำคนสูงสุด 5 คนต่อ 1 งาน

for _, row in df_matched.iterrows():
    task_name = row['task_name']
    sub_task = row['sub_task']
    matched_skill = row['matched_skill']
    start_date = row['start_date']  # ✅ ตรงนี้ถูกต้อง
    end_date = row['end_date']

    if matched_skill in skill_to_workers:
        candidates = skill_to_workers[matched_skill]

        # เรียงตามคะแนนประเมินจากมาก → น้อย
        candidates_sorted = sorted(
            candidates,
            key=lambda x: -x['evaluation_type_ratings']
        )

        # เลือก top 5
        selected_workers = candidates_sorted[:top]

        for worker in selected_workers:
            assigned_workers.append({
                "task_name": task_name,
                "sub_task": sub_task,
                "matched_skill": matched_skill,
                "start_date": start_date,
                "end_date": end_date,
                "worker_id": worker['worker_id'],
                "worker_name": worker['worker_name'],
                "evaluation_type_ratings": worker['evaluation_type_ratings']
            })
    else:
        assigned_workers.append({
            "task_name": task_name,
            "sub_task": sub_task,
            "matched_skill": matched_skill,
            "start_date": start_date,
            "end_date": end_date,
            "worker_id": "N/A",
            "worker_name": "ไม่สามารถจับคู่ได้",
            "evaluation_type_ratings": "N/A"
        })

# ---------- สร้าง DataFrame และบันทึก ----------
assigned_df = pd.DataFrame(assigned_workers)

print(assigned_df)

'''
assigned_df['start_date'] = pd.to_datetime(assigned_df['start_date'], errors='coerce')
assigned_df['end_date'] = pd.to_datetime(assigned_df['end_date'], errors='coerce')
assigned_df = assigned_df.sort_values(by='start_date')
print(assigned_df.head(10))  # แสดง 10 แถวแรกหลังเรียง
'''
assigned_df.to_csv("assigned_worker_per_task.csv", index=False)