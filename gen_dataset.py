import json
import os
import csv
import re
import random
from datetime import datetime, timedelta
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from typing import List, Dict
from sklearn.model_selection import train_test_split

# Choose the mode and dataset to process.
MODE = "train"
# DATA = "PMData"         # For LifeSnaps (PMData) dataset processing (existing code)
DATA = "AW_FB"            # For AW_FB dataset processing using the attached CSV files

# Define all subtasks to process
SUBTASKS = ['sleep_quality', 'stress', 'readiness', 'fatigue']

def avg(_list):
    return sum(_list) / len(_list)

def json_reader(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return data

def csv_reader(file_name):
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        return reader

def convert_to_lowercase(input_string):
    return input_string.lower()

def has_alphabets(input_string):
    return any(char.isalpha() for char in input_string)

def has_numbers(input_string):
    return any(char.isdigit() for char in input_string)

def extract_words_inside_brackets(input_string):
    pattern = r"\[\'([^\']+)\'\]|\[\"([^\"]+)\"\]"
    matches = re.findall(pattern, input_string)
    extracted_words = ' '.join([word for match in matches for word in match if word])
    return extracted_words

if MODE == "train":
    if DATA == "PMData":
        print("[INFO] Generating datasets for PMData (LifeSnaps) ...")
        """
            The original LifeSnaps processing code goes here.
            (For brevity, this branch is unchanged from your provided code.)
        """
        # Example: reading participant info from Excel and processing directories
        try:
            participant_df = pd.read_excel(r"C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\pmdata\participant-overview.xlsx")
            participant_info = {}
            for _, row in participant_df.iterrows():
                pid = f"p{row['Participant']:02d}"
                participant_info[pid] = [
                    row.get('Age', -1),
                    row.get('Height', -1),
                    row.get('Gender', 'N/A'),
                    row.get('Weight', 'N/A'),
                    row.get('Activity Level', 'N/A')
                ]
        except Exception as e:
            print(f"[WARNING] Could not read participant info: {e}")
            participant_info = {f'p{i:02d}': [-1, -1, 'N/A', 'N/A', 'N/A'] for i in range(1, 17)}

        DATA_PATH = r"C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\dataverse_files"  # Adjust path as needed

        for SUBTASK in SUBTASKS:
            final_data = []
            print(f"\n[INFO] Processing subtask: {SUBTASK}")
            for dir1 in tqdm(os.listdir(DATA_PATH)):
                if "." in dir1:
                    continue
                tmp = participant_info[dir1]
                age = tmp[0]
                height = str(tmp[1]) + " cm"
                gender = tmp[2]
                fpath1 = os.path.join(DATA_PATH, dir1)
                if '.' not in dir1:
                    for dir2 in os.listdir(fpath1):
                        fpath2 = os.path.join(fpath1, dir2)
                        if dir2 == 'fitbit':
                            exercise_data = json_reader(fpath2 + '/exercise.json')
                            try:
                                heart_rate_data = json_reader(fpath2 + '/resting_heart_rate.json')
                            except:
                                continue
                            sleep_data = json_reader(fpath2 + '/sleep.json')
                        elif dir2 == 'pmsys':
                            wellness_data = csv_reader(fpath2 + "/wellness.csv")
                            wellness_dict = {k: [] for k in ['effective_time_frame', 'fatigue', 'mood', 'readiness', 'sleep_duration_h', 'sleep_quality', 'soreness', 'soreness_area', 'stress']}
                            for i, data in enumerate(wellness_data):
                                if i == 0:
                                    continue
                                date = data[0][:10] + "_" + data[0][11:][:-1].split(".")[0]
                                fatigue = data[1]
                                mood = data[2]
                                readiness = data[3]
                                sleep_dur = data[4]
                                sleep_qual = data[5]
                                stress = data[-1]
                                wellness_dict['effective_time_frame'].append(date)
                                wellness_dict['fatigue'].append(fatigue)
                                wellness_dict['mood'].append(mood)
                                wellness_dict['readiness'].append(readiness)
                                wellness_dict['sleep_duration_h'].append(sleep_dur)
                                wellness_dict['sleep_quality'].append(sleep_qual)
                                wellness_dict['stress'].append(stress)
                date = wellness_dict['effective_time_frame']
                fatigue = wellness_dict['fatigue']
                mood = wellness_dict['mood']
                readiness = wellness_dict['readiness']
                sleep_dur = wellness_dict['sleep_duration_h']
                sleep_qual = wellness_dict['sleep_quality']
                stress = wellness_dict['stress']
                for d, f, m, r, sd, sq, s in zip(date, fatigue, mood, readiness, sleep_dur, sleep_qual, stress):
                    new_d = datetime.strptime(d, '%Y-%m-%d_%H:%M:%S')
                    # Process exercise history
                    exercise_hist = []
                    while True:
                        for e_data in exercise_data:
                            e_date = e_data['startTime'][:10] + "_" + e_data['startTime'][11:]
                            new_ed = datetime.strptime(e_date, '%Y-%m-%d_%H:%M:%S')
                            if (new_d > new_ed) and (new_d - new_ed) < timedelta(days=10):
                                try:
                                    activity = e_data['activityName']
                                    burn_calories = float(e_data['calories'])
                                    steps = float(e_data['steps'])
                                    duration = float(e_data['duration']) / 1000 / 60  # in minutes
                                    exercise_hist.append([new_ed, activity, duration, burn_calories, steps])
                                except:
                                    continue
                        break
                    # Process sleep history
                    sleep_hist = []
                    while True:
                        for s_data in sleep_data:
                            s_date = s_data['startTime'][:10] + "_" + s_data['startTime'][11:]
                            new_sd = datetime.strptime(s_date, '%Y-%m-%d_%H:%M:%S')
                            if (new_d > new_sd) and (new_d - new_sd) < timedelta(days=10):
                                sleep_duration = float(s_data['duration']) / 1000 / 60  # in minutes
                                sleep_hist.append([new_sd, sleep_duration])
                        break
                    # Process heart rate history
                    hr_hist = []
                    while True:
                        for hr_data in heart_rate_data:
                            hr_date = hr_data['dateTime'][:10] + "_" + hr_data['dateTime'][11:]
                            new_hrd = datetime.strptime(hr_date, '%Y-%m-%d_%H:%M:%S')
                            if (new_d > new_hrd) and (new_d - new_hrd) < timedelta(days=10):
                                rhr = float(hr_data['value']['value'])
                                hr_hist.append([new_hrd, rhr])
                        break
                    try:
                        steps_10d = sum([x[-1] for x in exercise_hist]) / len(exercise_hist)
                    except:
                        steps_10d = "N/A"
                        continue
                    try:
                        calories_10d = sum([x[-2] for x in exercise_hist]) / len(exercise_hist)
                    except:
                        calories_10d = "N/A"
                        continue
                    try:
                        rhr_10d = sum([x[-1] for x in hr_hist]) / len(hr_hist)
                    except:
                        rhr_10d = "N/A"
                        continue
                    try:
                        sleep_dur_10d = sum([x[-1] for x in sleep_hist]) / len(sleep_hist)
                    except:
                        sleep_dur_10d = "N/A"
                        continue
                    if SUBTASK == "readiness":
                        range1 = 0
                        range2 = 10
                    elif SUBTASK == "stress":
                        range1 = 1
                        range2 = 5
                    elif SUBTASK == "sleep_quality":
                        range1 = 1
                        range2 = 5
                    elif SUBTASK == "fatigue":
                        range1 = 1
                        range2 = 5
                    I = "You are a personalized healthcare agent trained to predict {} which ranges from {} to {} based on physiological data and user information.".format(SUBTASK, range1, range2)
                    Q = "The recent 10-days sensor readings show: [Steps]: {:.0f} steps/day, [Burned Calories]: {:.0f} calories/day, [Heart Rate]: {:.0f} beats/min, [SleepMinutes]: {:.0f} minutes, [Mood]: {} out of 5".format(steps_10d, calories_10d, rhr_10d, sleep_dur_10d, m)
                    if SUBTASK == "readiness":
                        A = "The predicted {} level is {}.".format(SUBTASK, r)
                    elif SUBTASK == "stress":
                        A = "The predicted {} level is {}.".format(SUBTASK, s)
                    elif SUBTASK == "sleep_quality":
                        A = "The predicted {} level is {}.".format(SUBTASK, sq)
                    elif SUBTASK == "fatigue":
                        A = "The predicted {} level is {}.".format(SUBTASK, f)
                    final_data.append({'instruction': I, 'input': Q, 'output': A})
            
            # Train/eval split for LifeSnaps
            N = len(final_data)
            final_train_data = []
            final_eval_data = []
            random.seed(123)
            random.shuffle(final_data)
            eval_idx = 1
            for n, fd in enumerate(final_data):
                if n < int(N * 0.8):
                    final_train_data.append(fd)
                else:
                    if eval_idx < 300:
                        fd['no'] = eval_idx
                        fd['question'] = fd['input']
                        fd['answer'] = fd['output']
                        del fd['input']
                        del fd['output']
                        final_eval_data.append(fd)
                        eval_idx += 1
            
            os.makedirs('data', exist_ok=True)
            os.makedirs(f'data/pmdata_{SUBTASK}', exist_ok=True)
            print(f"\n[INFO] Saving datasets for {SUBTASK}...")
            for n_shot in [3, 10, 25]:
                output_file = f"data/PMData_{SUBTASK}_train_{n_shot}.json"
                with open(output_file, "w") as f:
                    json.dump(final_train_data[:n_shot], f, indent=4)
                print(f"[INFO] Saved {n_shot}-shot training file: {output_file}")
            output_file = f"data/PMData_{SUBTASK}_train_all.json"
            with open(output_file, "w") as f:
                json.dump(final_train_data, f, indent=4)
            print(f"[INFO] Saved full training file: {output_file}")
            eval_file = f"data/pmdata_{SUBTASK}/eval.json"
            with open(eval_file, "w") as f:
                json.dump(final_eval_data, f, indent=4)
            print(f"[INFO] Saved evaluation file: {eval_file}")
            
            print(f"\n[INFO] Generated {len(final_train_data)} training samples and {len(final_eval_data)} evaluation samples for {SUBTASK}")
        
        print("\n[INFO] Dataset generation complete for all subtasks for PMData!")
    
    elif DATA == "AW_FB":
        print("[INFO] Generating datasets for AW_FB ...")
        # Load the attached CSV files
        try:
            df_aw_fb = pd.read_csv(r"C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\dataverse_files\aw_fb_data.csv")
            print("[INFO] Loaded aw_fb_data.csv successfully.")
        except Exception as e:
            print(f"[WARNING] Could not read aw_fb_data.csv: {e}")
            df_aw_fb = pd.DataFrame()
        
        try:
            df_weka_aw = pd.read_csv(r"C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\dataverse_files\data_for_weka_aw.csv")
            print("[INFO] Loaded data_for_weka_aw.csv successfully.")
        except Exception as e:
            print(f"[WARNING] Could not read data_for_weka_aw.csv: {e}")
            df_weka_aw = pd.DataFrame()
        
        try:
            df_weka_fb = pd.read_csv(r"C:\Users\SIU856561805\HealthLLMFinal\HealthLLMV1\dataverse_files\data_for_weka_fb.csv")
            print("[INFO] Loaded data_for_weka_fb.csv successfully.")
        except Exception as e:
            print(f"[WARNING] Could not read data_for_weka_fb.csv: {e}")
            df_weka_fb = pd.DataFrame()
        
        # Optionally merge additional CSVs based on a common key (for example, "id")
        if not df_aw_fb.empty:
            if not df_weka_aw.empty and "id" in df_aw_fb.columns and "id" in df_weka_aw.columns:
                df_aw_fb = df_aw_fb.merge(df_weka_aw, on="id", how="left")
                print("[INFO] Merged data_for_weka_aw.csv into aw_fb_data.csv based on 'id'.")
            if not df_weka_fb.empty and "id" in df_aw_fb.columns and "id" in df_weka_fb.columns:
                df_aw_fb = df_aw_fb.merge(df_weka_fb, on="id", how="left")
                print("[INFO] Merged data_for_weka_fb.csv into aw_fb_data.csv based on 'id'.")
        
        # Process each row to generate final dataset entries.
        # Here we assume that df_aw_fb contains aggregated sensor data with at least the following columns:
        # 'steps_10d', 'calories_10d', 'heart_rate_10d', 'sleep_duration_10d', 'mood',
        # 'readiness', 'stress', 'sleep_quality', and 'fatigue'.
        final_data = []
        for _, row in df_aw_fb.iterrows():
            steps_10d = row.get("steps_10d", 0)
            calories_10d = row.get("calories_10d", 0)
            heart_rate_10d = row.get("heart_rate_10d", 0)
            sleep_duration_10d = row.get("sleep_duration_10d", 0)
            mood = row.get("mood", 0)
            readiness = row.get("readiness", 0)
            stress = row.get("stress", 0)
            sleep_quality = row.get("sleep_quality", 0)
            fatigue = row.get("fatigue", 0)
            
            for SUBTASK in SUBTASKS:
                if SUBTASK == "readiness":
                    range1, range2, value = 0, 10, readiness
                elif SUBTASK == "stress":
                    range1, range2, value = 1, 5, stress
                elif SUBTASK == "sleep_quality":
                    range1, range2, value = 1, 5, sleep_quality
                elif SUBTASK == "fatigue":
                    range1, range2, value = 1, 5, fatigue
                
                instruction = f"You are a personalized healthcare agent trained to predict {SUBTASK} which ranges from {range1} to {range2} based on physiological data and user information."
                question = (f"The recent 10-days sensor readings show: [Steps]: {steps_10d} steps/day, "
                            f"[Burned Calories]: {calories_10d} calories/day, [Heart Rate]: {heart_rate_10d} beats/min, "
                            f"[SleepMinutes]: {sleep_duration_10d} minutes, [Mood]: {mood} out of 5")
                answer = f"The predicted {SUBTASK} level is {value}."
                final_data.append({"instruction": instruction, "input": question, "output": answer})
        
        # Create train/eval splits
        N = len(final_data)
        final_train_data = []
        final_eval_data = []
        random.seed(123)
        random.shuffle(final_data)
        eval_idx = 1
        for n, fd in enumerate(final_data):
            if n < int(N * 0.8):
                final_train_data.append(fd)
            else:
                if eval_idx < 300:
                    fd["no"] = eval_idx
                    fd["question"] = fd["input"]
                    fd["answer"] = fd["output"]
                    del fd["input"]
                    del fd["output"]
                    final_eval_data.append(fd)
                    eval_idx += 1
        
        # Create output directories
        os.makedirs("data", exist_ok=True)
        for SUBTASK in SUBTASKS:
            os.makedirs(f"data/pmdata_{SUBTASK}", exist_ok=True)
        
        # Save training and evaluation datasets for each subtask
        print(f"\n[INFO] Saving datasets for AW_FB...")
        for SUBTASK in SUBTASKS:
            train_data_sub = [entry for entry in final_train_data if SUBTASK in entry["instruction"]]
            for n_shot in [3, 10, 25]:
                output_file = f"data/AWFB_{SUBTASK}_train_{n_shot}.json"
                with open(output_file, "w") as f:
                    json.dump(train_data_sub[:n_shot], f, indent=4)
                print(f"[INFO] Saved {n_shot}-shot training file: {output_file}")
            output_file = f"data/AWFB_{SUBTASK}_train_all.json"
            with open(output_file, "w") as f:
                json.dump(train_data_sub, f, indent=4)
            print(f"[INFO] Saved full training file: {output_file}")
            eval_data_sub = [entry for entry in final_eval_data if SUBTASK in entry["instruction"]]
            eval_file = f"data/pmdata_{SUBTASK}/eval.json"
            with open(eval_file, "w") as f:
                json.dump(eval_data_sub, f, indent=4)
            print(f"[INFO] Saved evaluation file: {eval_file}")
        
        print(f"\n[INFO] Generated {len(final_train_data)} training samples and {len(final_eval_data)} evaluation samples for AW_FB")
        print("\n[INFO] Dataset generation complete for all subtasks for AW_FB!")
