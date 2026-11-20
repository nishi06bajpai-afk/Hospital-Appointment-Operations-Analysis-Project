"""
generate_dataset.py
Creates a realistic (synthetic) operational dataset: 09_hospital_appointments.csv
- 1,600 unique appointments + 12 deliberate duplicate rows (to practise Task 126)
- ~4% missing waiting times among attended visits (to practise Task 125)
- a few very long waits (outliers for Task 135)
No real patient data is used. Operational data only - no clinical information.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 1600

# department: (volume weight, mean wait, no-show base prob, fee min, fee max, doctors)
DEPTS = {
    "General Medicine": (0.22, 28, 0.14, 300, 600,  ["Dr. A. Sharma", "Dr. R. Verma", "Dr. S. Khan"]),
    "Cardiology":       (0.11, 42, 0.10, 800, 1500, ["Dr. P. Mehta", "Dr. N. Iyer"]),
    "Orthopedics":      (0.13, 38, 0.16, 600, 1200, ["Dr. V. Singh", "Dr. K. Reddy", "Dr. M. Bose"]),
    "Pediatrics":       (0.14, 25, 0.12, 400, 700,  ["Dr. L. Gupta", "Dr. T. Nair"]),
    "Gynecology":       (0.11, 35, 0.11, 600, 1100, ["Dr. S. Kapoor", "Dr. D. Joshi"]),
    "Dermatology":      (0.10, 22, 0.22, 500, 900,  ["Dr. H. Malik", "Dr. J. Das"]),
    "ENT":              (0.09, 30, 0.18, 400, 800,  ["Dr. B. Rao", "Dr. C. Pillai"]),
    "Neurology":        (0.10, 50, 0.09, 900, 1800, ["Dr. G. Menon", "Dr. F. Ali"]),
}
names = list(DEPTS)
weights = np.array([DEPTS[d][0] for d in names]); weights /= weights.sum()

dates = pd.date_range("2026-01-01", "2026-03-31")
dates = dates[dates.dayofweek != 6]                     # no Sundays
day_w = np.where(dates.dayofweek == 0, 1.35, 1.0)       # Mondays are busier
day_w = day_w / day_w.sum()

rows = []
for i in range(1, N + 1):
    dept = rng.choice(names, p=weights)
    _, wmean, ns_base, fmin, fmax, docs = DEPTS[dept]
    doctor = rng.choice(docs)
    date = rng.choice(dates, p=day_w)
    hour = int(rng.choice(range(9, 17), p=[.14, .16, .15, .12, .08, .12, .12, .11]))
    minute = int(rng.choice([0, 15, 30, 45]))
    atype = rng.choice(["New", "Follow-up"], p=[0.45, 0.55])
    sms = rng.choice(["Yes", "No"], p=[0.60, 0.40])
    age = int(np.clip(rng.normal(8 if dept == "Pediatrics" else 42, 5 if dept == "Pediatrics" else 18), 1, 90))
    if dept == "Gynecology":
        gender = "Female"
    else:
        gender = rng.choice(["Male", "Female"], p=[0.5, 0.5])

    # no-show / cancel probabilities (SMS reminder lowers no-show)
    p_ns = ns_base * (0.55 if sms == "Yes" else 1.6) * (0.85 if atype == "Follow-up" else 1.1)
    p_ns = min(p_ns, 0.6)
    p_cx = 0.06
    r = rng.random()
    status = "No-Show" if r < p_ns else ("Cancelled" if r < p_ns + p_cx else "Attended")

    wait = np.nan
    if status == "Attended":
        base = wmean * (0.78 if atype == "Follow-up" else 1.15)
        peak = 1.15 if hour in (10, 11) else 1.0
        wait = max(2, rng.gamma(shape=4, scale=base * peak / 4))
        if rng.random() < 0.02:                         # extreme waits (outliers)
            wait += rng.integers(70, 140)
        wait = int(round(wait))

    fee_base = rng.integers(fmin // 50, fmax // 50 + 1) * 50
    fee = int(fee_base * (0.6 if atype == "Follow-up" else 1.0) // 50 * 50)
    fee = max(fee, 200)

    rows.append([f"APT{i:05d}", f"PAT{rng.integers(1000, 1500):04d}", age, gender, dept, doctor,
                 pd.Timestamp(date).strftime("%Y-%m-%d"), f"{hour:02d}:{minute:02d}",
                 atype, sms, status, wait, fee])

cols = ["Appointment_ID", "Patient_ID", "Patient_Age", "Gender", "Department", "Doctor_Name",
        "Appointment_Date", "Appointment_Time", "Appointment_Type", "SMS_Reminder",
        "Attendance_Status", "Waiting_Time_Min", "Consultation_Fee"]
df = pd.DataFrame(rows, columns=cols)

# missing waiting times among ATTENDED visits (data-entry gaps) ~4%
att_idx = df.index[df["Attendance_Status"] == "Attended"]
miss = rng.choice(att_idx, size=int(len(att_idx) * 0.04), replace=False)
df.loc[miss, "Waiting_Time_Min"] = np.nan

# 12 duplicate Appointment_ID rows (copied records)
dups = df.sample(12, random_state=7)
df = pd.concat([df, dups]).sample(frac=1, random_state=3).reset_index(drop=True)
df = df.sort_values("Appointment_ID", kind="stable").reset_index(drop=True)

df.to_csv("09_hospital_appointments.csv", index=False)
print("Saved", df.shape, "-> 09_hospital_appointments.csv")
