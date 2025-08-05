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
# ---------- จับคู่ 1 งาน → คนที่มีทักษะตรง ----------
assigned_workers = []
top = 5
used_worker_names = set()

assigned_workers = []
top = 5
used_worker_names = set()

for _, row in df_matched.iterrows():
    task_name = row['task_name']
    sub_task = row['sub_task']
    matched_skills = str(row['matched_skill']).split(',')
    matched_skills = [skill.strip() for skill in matched_skills if skill.strip()]
    start_date = row['start_date']
    end_date = row['end_date']

    for skill in matched_skills:
        if skill in skill_to_workers:
            candidates = skill_to_workers[skill]

            candidates_sorted = sorted(
                candidates,
                key=lambda x: -x['evaluation_type_ratings']
            )

            count = 0
            for worker in candidates_sorted:
                if worker['worker_name'] not in used_worker_names:
                    assigned_workers.append({
                        "task_name": task_name,
                        "sub_task": sub_task,
                        "matched_skill": skill,  # ✅ ใช้ skill เดี่ยวเท่านั้น
                        "start_date": start_date,
                        "end_date": end_date,
                        "worker_id": worker['worker_id'],
                        "worker_name": worker['worker_name'],
                        "evaluation_type_ratings": worker['evaluation_type_ratings']
                    })
                    used_worker_names.add(worker['worker_name'])
                    count += 1
                if count >= top:
                    break
        else:
            assigned_workers.append({
                "task_name": task_name,
                "sub_task": sub_task,
                "matched_skill": skill,
                "start_date": start_date,
                "end_date": end_date,
                "worker_id": "N/A",
                "worker_name": "ไม่สามารถจับคู่ได้",
                "evaluation_type_ratings": "N/A"
            })


# ---------- สร้าง DataFrame และคำนวณวันที่ ----------
assigned_df = pd.DataFrame(assigned_workers)

# แปลงวันที่
assigned_df['start_date'] = pd.to_datetime(assigned_df['start_date'], format='%Y-%m-%d', errors='coerce')
assigned_df['end_date'] = pd.to_datetime(assigned_df['end_date'], format='%Y-%m-%d', errors='coerce')
assigned_df['duration_days'] = (assigned_df['end_date'] - assigned_df['start_date']).dt.days

# ส่งออก CSV
assigned_df.to_csv("assigned_worker_per_task1.csv", index=False)
