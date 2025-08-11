import os
import re
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from collections import defaultdict

# -------------------- ฟังก์ชันแปลง Skill Level --------------------
def parse_skill_level(val):
    """แปลงค่าระดับทักษะ เช่น 'Level 1U' → (1.0, 'U')"""
    text = str(val).strip()

    # ถ้าเป็นคำที่สื่อถึงทดลองงาน → 0.0 และ suffix 'U'
    if re.search(r'ทดลอง|ฝึกงาน|intern', text, re.IGNORECASE):
        return 0.0, 'U'

    # ดึงตัวเลข เช่น "1", "1.5"
    num_match = re.search(r'(\d+(\.\d+)?)', text)
    number = float(num_match.group(1)) if num_match else None

    # ดึง suffix หลังตัวเลข เช่น "U", "A"
    suffix = ''
    if num_match:
        after_num = text[num_match.end():].strip()
        suffix_match = re.match(r'([A-Za-z]+)', after_num)
        if suffix_match:
            suffix = suffix_match.group(1)

    return number, suffix


def clean_skill_level(df_skills, col_name='level_code'):
    """เพิ่มคอลัมน์ skill_level_num และ skill_level_suffix"""
    parsed = df_skills[col_name].apply(parse_skill_level)
    df_skills = df_skills.copy()
    df_skills['skill_level_num'] = parsed.apply(lambda x: x[0])
    df_skills['skill_level_suffix'] = parsed.apply(lambda x: x[1])

    # กรองแถวที่มีค่าตัวเลข
    return df_skills[df_skills['skill_level_num'].notna()]


# -------------------- STEP 0: Load Employee & Position --------------------
try:
    sfc_employ = pd.read_csv('sfc_employee.csv')
    position = pd.read_csv('position - sfc_position_202508071504.csv')
    tasks = pd.read_csv('tasks.csv')
    subtasks = pd.read_csv('subtasks.csv')
    df_stand = pd.read_csv('work_standards.csv')
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    exit(1)

# เลือกคอลัมน์ที่ต้องใช้จาก sfc_employee
work_sfc = sfc_employ[['empcode', 'empfullname_th', 'age',
                       'position_code', 'level_code',
                       'amount_day', 'amount_hours']]

# แปลง level_code
work_sfc = clean_skill_level(work_sfc, col_name='level_code')
print("✅ Work SFC after level transform:")
print(work_sfc.head())

# รวมข้อมูลพนักงานกับชื่อตำแหน่ง
work_sfc_full = pd.merge(
    work_sfc,
    position[['position_code', 'position_name']],
    on='position_code',
    how='left'
)
print("✅ Position table:")
print(position.head())

# ลบคอลัมน์ที่ไม่ใช้จาก tasks
tasks = tasks.drop(['worker_id', 'project_id'], axis=1, errors='ignore')

# รวม tasks และ subtasks
tasks_full = pd.merge(
    tasks,
    subtasks,
    on="task_id",
    how="left"
)
print("✅ Tasks full:")
print(tasks_full.head())

# บันทึก tasks_full
tasks_full.to_csv('tasks_full.csv', index=False, encoding='utf-8')

# -------------------- Load API Key --------------------
load_dotenv()
API_KEY = os.getenv("gemini_api_key")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# -------------------- STEP 1: Match Skill to Tasks --------------------
df_worker_skill = work_sfc_full['position_name'].dropna().unique().tolist()

task_list = [
    f"- งาน: {str(row['task_name']).strip()}, รายละเอียดงานย่อย: {str(row['subtask']).strip()}, "
    f"ประเภทงานย่อย: {str(row['sub_task_type']).strip()}, ทักษะที่ต้องการ: {str(row['skill_type_id']).strip()}"
    for _, row in tasks_full.iterrows()
]

prompt_skill = f"""
คุณเป็นผู้เชี่ยวชาญด้านการจับคู่ทักษะของแรงงานกับงานที่ต้องทำ
โดยมีทักษะของแรงงานดังต่อไปนี้:
{', '.join(df_worker_skill)}   

และรายการงานที่ต้องการคนทำดังนี้:
{chr(10).join(task_list)}

กรุณาจับคู่ให้ว่าแต่ละงานควรใช้ทักษะใดจากรายการที่มีมากที่สุดโดยที่พยามหาความใกล้เคียงด้วย
ถ้าไม่สามารถจับคู่ได้ ให้ตอบว่า "ไม่สามารถจับคู่ได้"

แสดงผลในรูปแบบตารางที่มี 3 คอลัมน์: 
- ชื่องาน
- งานย่อย
- ทักษะที่จับคู่ได้
"""

response_skill = model.generate_content(prompt_skill)

# -------------------- แปลงผล Gemini เป็น DataFrame --------------------
lines = response_skill.text.split('\n')
results = []
header_found = False
for line in lines:
    if line.strip() and '|' in line and not line.startswith('|-'):
        if not header_found and 'ชื่องาน' in line:
            header_found = True
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 3:
            task_name, sub_task, skill = parts[:3]
            match_row = tasks_full[
                (tasks_full['task_name'].astype(str).str.strip() == task_name) &
                (tasks_full['subtask'].astype(str).str.strip() == sub_task)
            ]
            if not match_row.empty:
                row = match_row.iloc[0]
                results.append({
                    "task_id": row.get('task_id', None),
                    "task_name": task_name,
                    "subtask_id": row.get('subtask_id', None),
                    "sub_task": sub_task,
                    "matched_skill": skill,
                    "contract_amount": row.get('contract_amount', None),
                    "contract_quantity": row.get('contract_quantity', None),
                    "weight": row.get('weight', None),
                    "plan_amount": row.get('plan_amount', None),
                    "plan_progress": row.get('plan_progress', None),
                    "durations_x": row.get('durations_x', None),
                    "durations_y": row.get('durations_y', None),
                    "skill_type_id": row.get('skill_type_id', None),
                    "skill_record_id": row.get('skill_record_id', None),
                    "sub_task_type": row.get('sub_task_type', None),
                    "required_worker": row.get('required_worker', None)
                })

df_matched = pd.DataFrame(results)
df_matched['durations_x'] = pd.to_numeric(df_matched['durations_x'], errors='coerce')
df_matched['durations_y'] = pd.to_numeric(df_matched['durations_y'], errors='coerce')
df_matched['duration_days'] = df_matched.apply(
    lambda row: row['durations_y'] if pd.notna(row['durations_y']) else row['durations_x'], axis=1
)


# -------------------- STEP 2: Match Work Standard to Tasks --------------------
df_stand_list = df_stand['task'].dropna().astype(str).str.strip().str.lower().unique().tolist()
task_list = [
    f"- งาน: {str(row['task_name']).strip()}, รายละเอียดงานย่อย: {str(row['sub_task']).strip()}"
    for _, row in df_matched.iterrows()
]

prompt_std = f"""
คุณเป็นผู้เชี่ยวชาญด้านการจับคู่งานเข้ากับมาตรฐานการทำงาน
โดยมีมาตรฐานการทำงานดังต่อไปนี้:
{', '.join(df_stand_list)}

และรายการงานที่ต้องการคนทำดังนี้:
{chr(10).join(task_list)}

กรุณาจับคู่ให้ว่าแต่ละงานควรใช้มาตรฐานใดจากรายการที่มีมากที่สุดโดยที่พยามหาความใกล้เคียงด้วย
ถ้าไม่สามารถจับคู่ได้ ให้ตอบว่า "ไม่สามารถจับคู่ได้"

แสดงผลในรูปแบบตารางที่มี 3 คอลัมน์:
- ชื่องาน
- งานย่อย
- มาตรฐานที่จับคู่ได้
"""

response_std = model.generate_content(prompt_std)
results = []
for line in response_std.text.split('\n'):
    line = line.strip()
    if not line or "ชื่องาน" in line or line.startswith('|-'):
        continue
    parts = [p.strip() for p in line.split('|') if p.strip()]
    if len(parts) >= 3:
        results.append({
            'task_name': parts[0].lower(),
            'sub_task': parts[1].lower(),
            'matched_standard': parts[2].lower()
        })

df_std_match = pd.DataFrame(results)
df_matched[['task_name', 'sub_task']] = df_matched[['task_name', 'sub_task']].apply(lambda x: x.astype(str).str.lower().str.strip())
df_stand['task'] = df_stand['task'].astype(str).str.lower().str.strip()

df_final = df_matched.merge(df_std_match, on=['task_name', 'sub_task'], how='left')
df_final = df_final.merge(df_stand[['task', 'standard_rate']], how='left', left_on='matched_standard', right_on='task')
df_final.drop(columns=['task'], inplace=True)
df_final['contract_quantity'] = pd.to_numeric(df_final['contract_quantity'], errors='coerce')
df_final['standard_rate'] = pd.to_numeric(df_final['standard_rate'], errors='coerce')
df_final['required_worker'] = pd.to_numeric(df_final['required_worker'], errors='coerce')
df_final['workers_needed'] = df_final.apply(
    lambda row: int(row['required_worker']) if pd.notna(row['required_worker']) else 
                int(np.ceil(row['contract_quantity'] / (row['standard_rate'] * row['duration_days']))) 
                if row['standard_rate'] > 0 and row['duration_days'] > 0 else 5,
    axis=1
)






# -------------------- STEP 3: Assign Workers to Tasks --------------------
# ใช้ work_sfc_full แทน df_worker
df_worker = work_sfc_full[['empcode', 'empfullname_th', 'position_name', 'skill_level_num', 'skill_level_suffix']].dropna()
df_worker['evaluation_type_ratings'] = df_worker['skill_level_num']  # สมมติว่า skill_level_num เป็นคะแนนประเมิน
df_worker = df_worker.dropna(subset=['evaluation_type_ratings'])

skill_to_workers = defaultdict(list)
for _, row in df_worker.iterrows():
    skill = row['position_name'].strip()
    skill_to_workers[skill].append({
        "worker_id": row['empcode'],
        "worker_name": row['empfullname_th'],
        "skill_level": row['skill_level_num'],
        "evaluation_type_ratings": row['evaluation_type_ratings']
    })

assigned_workers = []
used_worker_names = set()

for _, row in df_final.iterrows():
    matched_skills = str(row['matched_skill']).split(',')
    matched_skills = [skill.strip() for skill in matched_skills if skill.strip()]
    count = 0
    for skill in matched_skills:
        if skill in skill_to_workers:
            candidates = sorted(skill_to_workers[skill], key=lambda x: -x['evaluation_type_ratings'])
            for worker in candidates:
                if worker['worker_name'] not in used_worker_names:
                    assigned_workers.append({
                        "task_id": row['task_id'],
                        "task_name": row['task_name'],
                        "subtask_id": row['subtask_id'],
                        "sub_task": row['sub_task'],
                        "matched_skill": skill,
                        "contract_amount": row['contract_amount'],
                        "contract_quantity": row['contract_quantity'],
                        "weight": row['weight'],
                        "plan_amount": row['plan_amount'],
                        "plan_progress": row['plan_progress'],
                        "duration_days": row['duration_days'],
                        "skill_type_id": row['skill_type_id'],
                        "skill_record_id": row['skill_record_id'],
                        "sub_task_type": row['sub_task_type'],
                        "required_worker": row['required_worker'],
                        "worker_id": worker['worker_id'],
                        "worker_name": worker['worker_name'],
                        "evaluation_type_ratings": worker['evaluation_type_ratings']
                    })
                    used_worker_names.add(worker['worker_name'])
                    count += 1
                if count >= int(row['workers_needed']):
                    break
        else:
            assigned_workers.append({
                "task_id": row['task_id'],
                "task_name": row['task_name'],
                "subtask_id": row['subtask_id'],
                "sub_task": row['sub_task'],
                "matched_skill": skill,
                "contract_amount": row['contract_amount'],
                "contract_quantity": row['contract_quantity'],
                "weight": row['weight'],
                "plan_amount": row['plan_amount'],
                "plan_progress": row['plan_progress'],
                "duration_days": row['duration_days'],
                "skill_type_id": row['skill_type_id'],
                "skill_record_id": row['skill_record_id'],
                "sub_task_type": row['sub_task_type'],
                "required_worker": row['required_worker'],
                "worker_id": "N/A",
                "worker_name": "ไม่สามารถจับคู่ได้",
                "evaluation_type_ratings": "N/A"
            })

assigned_df = pd.DataFrame(assigned_workers)



# ทำการรวมข้อมูลหา PF 

new_productivity = pd.read_csv('new_productivity_record.csv')

pf_df = pd.merge(
    df_final,
    new_productivity,
    on=["task_id", "subtask_id"],  # ใช้ทั้ง task_id และ subtask_id เพื่อแมตช์แม่นยำ
    how="left"
)


pf_df.to_csv('productivity_final.csv', index=False, encoding='utf-8')