import pandas as pd
import random
import os

def create_dataset(path, n=2000, seed=0):
    random.seed(seed)
    rows = []
    for i in range(n):
        attendance = random.randint(50, 100)
        prior_grades = random.uniform(0, 10)
        study_hours = random.uniform(0, 5)
        participation = random.randint(0, 10)
        assignments = random.randint(0, 10)
        stress = random.randint(1, 5)
        sleep = random.uniform(4, 9)
        label = 'High' if (attendance > 75 and prior_grades > 6 and study_hours > 2) else 'Low'
        rows.append({
            'attendance': attendance,
            'prior_grades': round(prior_grades,2),
            'study_hours': round(study_hours,2),
            'participation': participation,
            'assignments': assignments,
            'stress_level': stress,
            'sleep_hours': round(sleep,2),
            'performance': label
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

if __name__ == '__main__':
    create_dataset('data/student_data.csv', n=2000, seed=0)
    df = pd.read_csv('data/student_data.csv')
    sample = df.sample(10, random_state=1).drop(columns=['performance'])
    sample.to_csv('data/sample_input.csv', index=False)
    print('data/student_data.csv and data/sample_input.csv created')
