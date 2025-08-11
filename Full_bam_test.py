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
    if re.search(r'ทดลอง|ฝึกงาน|intern', text, re.IGNORECASE):
        return 0.0, 'U'
    num_match = re.search(r'(\d+(\.\d+)?)', text)
    number = float(num_match.group(1)) if num_match else None
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
    return df_skills[df_skills['skill_level_num'].notna()]

# -------------------- STEP 0: Load Data --------------------
try:
    sfc_employ = pd.read_csv('sfc_employee.csv')
    position = pd.read_csv('position - sfc_position_202508071504.csv')
    tasks = pd.read_csv('tasks.csv')
    subtasks = pd.read_csv('subtasks.csv')
    df_stand = pd.read_csv('work_standards.csv')
    new_productivity = pd.read_csv('new_productivity_record.csv')
except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    exit(1)

# เลือกคอลัมน์ที่ต้องใช้จาก sfc_employee
work_sfc = sfc_employ[['empcode', 'empfullname_th', 'age', 'position_code', 'level_code', 'amount_day', 'amount_hours']]
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

# -------------------- PART 1: Initial Planning and Worker Assignment --------------------
# STEP 1: Match Skill to Tasks (LLM)
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

# STEP 2: Match Work Standard to Tasks (LLM)
df_stand_list = df_stand['task'].dropna().astype(str).str.strip().str.lower().unique().tolist()
task_list = [
    f"- งาน: {str(row['task_name']).strip()}, รายละเอียดงานย่อย: {str(row['sub_task']).strip()}"
    for _, row in df_matched.iterrows()
]

prompt_std = f"""
คุณเป็นผู้เชี่ยวชาญด้านการจับคู่งานเข้กับมาตรฐานการทำงาน
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

# STEP 3: Assign Workers to Tasks (Initial)
df_worker = work_sfc_full[['empcode', 'empfullname_th', 'position_name', 'skill_level_num', 'skill_level_suffix']].dropna()
df_worker['evaluation_type_ratings'] = df_worker['skill_level_num']
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

initial_assigned_workers = []
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
                    initial_assigned_workers.append({
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
            initial_assigned_workers.append({
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

initial_assigned_df = pd.DataFrame(initial_assigned_workers)
initial_assigned_df.to_csv("initial_assigned_workers.csv", index=False, encoding='utf-8')
print("\u2705 initial_assigned_workers.csv saved")

# -------------------- PART 2: Update with PF and Reassign for Delayed Tasks --------------------
# รวมข้อมูลหา PF
pf_df = pd.merge(
    df_final,
    new_productivity,
    on=["task_id", "subtask_id"],
    how="left"
)

# คำนวณ PF = estimated_rate / actual_rate
pf_df['calculated_pf'] = pf_df['estimated_rate'] / pf_df['actual_rate']

# ระบุงานที่ล่าช้า (PF < 1)
delayed_tasks = pf_df[pf_df['calculated_pf'] < 1].copy()
delayed_tasks['remaining_quantity'] = delayed_tasks['contract_quantity'] - delayed_tasks['unit_completed']
delayed_tasks['remaining_days'] = delayed_tasks['durations_y'] - delayed_tasks['day_do']
delayed_tasks['remaining_days'] = delayed_tasks['remaining_days'].apply(lambda x: max(x, 1))
delayed_tasks['workers_needed_new'] = delayed_tasks.apply(
    lambda row: int(np.ceil(row['remaining_quantity'] / (row['actual_rate'] * 8 * row['remaining_days']))) if row['actual_rate'] > 0 else 5,
    axis=1
)
delayed_tasks['workers_to_add'] = delayed_tasks.apply(
    lambda row: max(row['workers_needed_new'] - row['required_worker'], 0) if pd.notna(row['required_worker']) else row['workers_needed_new'],
    axis=1
)
print("✅ Delayed tasks (PF < 1) - Lagged Work:")
print(delayed_tasks[['task_id', 'subtask_id', 'task_name', 'sub_task', 'calculated_pf', 'day_do', 'remaining_quantity', 'remaining_days', 'workers_needed_new', 'workers_to_add']].head())

# STEP 4: Match Skill for Delayed Tasks (LLM)
task_list_delayed = [
    f"- งาน: {str(row['task_name']).strip()}, รายละเอียดงานย่อย: {str(row['sub_task']).strip()}, "
    f"ประเภทงานย่อย: {str(row['sub_task_type']).strip()}, ทักษะที่ต้องการ: {str(row['skill_type_id']).strip()}, "
    f"Productivity Factor: {str(row['calculated_pf']).strip()}, "
    f"วันที่ทำไปแล้ว: {str(row['day_do']).strip()}, วันที่เหลือ: {str(row['remaining_days']).strip()}"
    for _, row in delayed_tasks.iterrows()
]

prompt_skill_delayed = f"""
คุณเป็นผู้เชี่ยวชาญด้านการจับคู่ทักษะของแรงงานกับงานที่ต้องทำ
โดยมีทักษะของแรงงานดังต่อไปนี้:
{', '.join(df_worker_skill)}

และรายการงานที่ล่าช้า (Productivity Factor < 1) ดังนี้:
{chr(10).join(task_list_delayed)}

กรุณาจับคู่ให้ว่าแต่ละงานควรใช้ทักษะใดจากรายการที่มีมากที่สุด โดยพิจารณาความใกล้เคียงของทักษะและให้ความสำคัญกับงานที่มี Productivity Factor ต่ำ เพื่อเลือกทักษะที่ช่วยเพิ่มประสิทธิภาพ
ถ้าไม่สามารถจับคู่ได้ ให้ตอบว่า "ไม่สามารถจับคู่ได้"

แสดงผลในรูปแบบตารางที่มี 3 คอลัมน์: 
- ชื่องาน
- งานย่อย
- ทักษะที่จับคู่ได้
"""

response_skill_delayed = model.generate_content(prompt_skill_delayed)

# แปลงผล Gemini เป็น DataFrame
lines = response_skill_delayed.text.split('\n')
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
            match_row = delayed_tasks[
                (delayed_tasks['task_name'].astype(str).str.strip() == task_name) &
                (delayed_tasks['sub_task'].astype(str).str.strip() == sub_task)
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
                    "required_worker": row.get('required_worker', None),
                    "calculated_pf": row.get('calculated_pf', None),
                    "day_do": row.get('day_do', None),
                    "remaining_quantity": row.get('remaining_quantity', None),
                    "remaining_days": row.get('remaining_days', None),
                    "workers_needed_new": row.get('workers_needed_new', None),
                    "workers_to_add": row.get('workers_to_add', None)
                })

df_matched_delayed = pd.DataFrame(results)
df_matched_delayed['durations_x'] = pd.to_numeric(df_matched_delayed['durations_x'], errors='coerce')
df_matched_delayed['durations_y'] = pd.to_numeric(df_matched_delayed['durations_y'], errors='coerce')
df_matched_delayed['calculated_pf'] = pd.to_numeric(df_matched_delayed['calculated_pf'], errors='coerce')
df_matched_delayed['remaining_quantity'] = pd.to_numeric(df_matched_delayed['remaining_quantity'], errors='coerce')
df_matched_delayed['remaining_days'] = pd.to_numeric(df_matched_delayed['remaining_days'], errors='coerce')
df_matched_delayed['workers_needed_new'] = pd.to_numeric(df_matched_delayed['workers_needed_new'], errors='coerce')
df_matched_delayed['workers_to_add'] = pd.to_numeric(df_matched_delayed['workers_to_add'], errors='coerce')
df_matched_delayed['duration_days'] = df_matched_delayed.apply(
    lambda row: row['durations_y'] if pd.notna(row['durations_y']) else row['durations_x'], axis=1
)
print("✅ Matched skills for delayed tasks:")
print(df_matched_delayed[['task_id', 'subtask_id', 'task_name', 'sub_task', 'matched_skill', 'calculated_pf']].head())

# STEP 5: Match Work Standard for Delayed Tasks (LLM)
task_list_delayed_std = [
    f"- งาน: {str(row['task_name']).strip()}, รายละเอียดงานย่อย: {str(row['sub_task']).strip()}, "
    f"Productivity Factor: {str(row['calculated_pf']).strip()}, "
    f"วันที่ทำไปแล้ว: {str(row['day_do']).strip()}, วันที่เหลือ: {str(row['remaining_days']).strip()}"
    for _, row in df_matched_delayed.iterrows()
]

prompt_std_delayed = f"""
คุณเป็นผู้เชี่ยวชาญด้านการจับคู่งานเข้กับมาตรฐานการทำงาน
โดยมีมาตรฐานการทำงานดังต่อไปนี้:
{', '.join(df_stand_list)}

และรายการงานที่ล่าช้า (Productivity Factor < 1) ดังนี้:
{chr(10).join(task_list_delayed_std)}

กรุณาจับคู่ให้ว่าแต่ละงานควรใช้มาตรฐานใดจากรายการที่มีมากที่สุด โดยพิจารณาความใกล้เคียงและให้ความสำคัญกับงานที่มี Productivity Factor ต่ำ เพื่อเลือกมาตรฐานที่ช่วยเพิ่มประสิทธิภาพ
ถ้าไม่สามารถจับคู่ได้ ให้ตอบว่า "ไม่สามารถจับคู่ได้"

แสดงผลในรูปแบบตารางที่มี 3 คอลัมน์:
- ชื่องาน
- งานย่อย
- มาตรฐานที่จับคู่ได้
"""

response_std_delayed = model.generate_content(prompt_std_delayed)
results = []
for line in response_std_delayed.text.split('\n'):
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

df_std_match_delayed = pd.DataFrame(results)
df_matched_delayed[['task_name', 'sub_task']] = df_matched_delayed[['task_name', 'sub_task']].apply(lambda x: x.astype(str).str.lower().str.strip())

df_final_delayed = df_matched_delayed.merge(df_std_match_delayed, on=['task_name', 'sub_task'], how='left')
df_final_delayed = df_final_delayed.merge(df_stand[['task', 'standard_rate']], how='left', left_on='matched_standard', right_on='task')
df_final_delayed.drop(columns=['task'], inplace=True)
df_final_delayed['contract_quantity'] = pd.to_numeric(df_final_delayed['contract_quantity'], errors='coerce')
df_final_delayed['standard_rate'] = pd.to_numeric(df_final_delayed['standard_rate'], errors='coerce')
df_final_delayed['required_worker'] = pd.to_numeric(df_final_delayed['required_worker'], errors='coerce')
df_final_delayed['workers_needed_new'] = df_final_delayed.apply(
    lambda row: int(np.ceil(row['remaining_quantity'] / (row['standard_rate'] * row['remaining_days']))) if row['standard_rate'] > 0 and row['remaining_days'] > 0 else 5,
    axis=1
)
df_final_delayed['workers_to_add'] = df_final_delayed.apply(
    lambda row: max(row['workers_needed_new'] - row['required_worker'], 0) if pd.notna(row['required_worker']) else row['workers_needed_new'],
    axis=1
)
print("✅ Final matched with standards for delayed tasks:")
print(df_final_delayed[['task_id', 'subtask_id', 'task_name', 'sub_task', 'matched_skill', 'matched_standard', 'workers_needed_new', 'workers_to_add']].head())

# STEP 6: Assign New Workers to Delayed Tasks
assigned_workers_delayed = []
used_worker_names_delayed = set()  # รีเซ็ตชุดชื่อพนักงานสำหรับส่วนที่สอง

for _, row in df_final_delayed.iterrows():
    matched_skills = str(row['matched_skill']).split(',')
    matched_skills = [skill.strip() for skill in matched_skills if skill.strip()]
    count = 0
    candidates = []
    for skill in matched_skills:
        if skill in skill_to_workers:
            candidates.extend(sorted(skill_to_workers[skill], key=lambda x: -x['evaluation_type_ratings']))

    for worker in candidates:
        if worker['worker_name'] not in used_worker_names_delayed:
            assigned_workers_delayed.append({
                "task_id": row['task_id'],
                "task_name": row['task_name'],
                "subtask_id": row['subtask_id'],
                "sub_task": row['sub_task'],
                "matched_skill": row['matched_skill'],
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
                "calculated_pf": row['calculated_pf'],
                "day_do": row['day_do'],
                "remaining_quantity": row['remaining_quantity'],
                "remaining_days": row['remaining_days'],
                "workers_needed_new": row['workers_needed_new'],
                "workers_to_add": row['workers_to_add'],
                "worker_id": worker['worker_id'],
                "worker_name": worker['worker_name'],
                "evaluation_type_ratings": worker['evaluation_type_ratings']
            })
            used_worker_names_delayed.add(worker['worker_name'])
            count += 1
        if count >= int(row['workers_to_add']):
            break

    if count < int(row['workers_to_add']):
        for _ in range(int(row['workers_to_add']) - count):
            assigned_workers_delayed.append({
                "task_id": row['task_id'],
                "task_name": row['task_name'],
                "subtask_id": row['subtask_id'],
                "sub_task": row['sub_task'],
                "matched_skill": row['matched_skill'],
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
                "calculated_pf": row['calculated_pf'],
                "day_do": row['day_do'],
                "remaining_quantity": row['remaining_quantity'],
                "remaining_days": row['remaining_days'],
                "workers_needed_new": row['workers_needed_new'],
                "workers_to_add": row['workers_to_add'],
                "worker_id": "N/A",
                "worker_name": "ไม่สามารถจับคู่ได้",
                "evaluation_type_ratings": "N/A"
            })

assigned_df_delayed = pd.DataFrame(assigned_workers_delayed)
assigned_df_delayed.to_csv("reassigned_workers_for_delayed_tasks.csv", index=False, encoding='utf-8')
print("\u2705 reassigned_workers_for_delayed_tasks.csv saved")