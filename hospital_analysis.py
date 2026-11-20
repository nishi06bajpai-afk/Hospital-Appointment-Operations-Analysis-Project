"""
Project 9: Hospital Appointment Operations Analysis
----------------------------------------------------
Menu-driven Python program (pandas + NumPy + if/else + loops) that analyses
appointment attendance, waiting times and department workload.

NOTE: This is an OPERATIONAL dataset. No medical conclusions are drawn.

Run:   python hospital_analysis.py            (interactive menu)
       python hospital_analysis.py --all      (runs every task, saves report files)
"""
import os
import sys
import numpy as np
import pandas as pd

DATA_FILE = "09_hospital_appointments.csv"
OUT_DIR = "output"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def heading(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def save(df, name):
    """Save a result table into the output folder (creates folder if needed)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path)
    print(f"[saved] {path}")


def pause():
    if INTERACTIVE:
        input("\nPress Enter to return to the menu...")


# ----------------------------------------------------------------------------
# Tasks 124-126 : load, inspect, missing values, duplicates
# ----------------------------------------------------------------------------
def load_data():
    """Task 124 - Load and inspect appointment records."""
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: '{DATA_FILE}' not found in this folder.")
        print("Run  python generate_dataset.py  first, or place the CSV here.")
        sys.exit(1)
    df = pd.read_csv(DATA_FILE)
    return df


def inspect_data(df):
    heading("TASK 124 - LOAD AND INSPECT APPOINTMENT RECORDS")
    print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}\n")
    print("First 5 records:")
    print(df.head())
    print("\nColumn data types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nNumeric summary:")
    print(df.describe())


def handle_missing_waiting(df):
    """
    Task 125 - Identify and handle missing waiting times.
    Logic:
      * No-Show / Cancelled  -> patient never saw a doctor, so waiting time is
        'not applicable'. We keep it blank (NaN) and do NOT fill it.
      * Attended but blank   -> data-entry gap. Fill with the MEDIAN waiting time
        of the same Department + Appointment_Type (median resists outliers).
    """
    heading("TASK 125 - MISSING WAITING TIMES")
    total_missing = df["Waiting_Time_Min"].isnull().sum()
    not_applicable = df[(df["Waiting_Time_Min"].isnull()) &
                        (df["Attendance_Status"] != "Attended")].shape[0]
    gaps = df[(df["Waiting_Time_Min"].isnull()) &
              (df["Attendance_Status"] == "Attended")].index
    print(f"Total missing waiting times        : {total_missing}")
    print(f"  Not applicable (No-Show/Cancel)  : {not_applicable}")
    print(f"  True data gaps (Attended visits) : {len(gaps)}")

    group_median = df.groupby(["Department", "Appointment_Type"])["Waiting_Time_Min"].transform("median")
    df["Wait_Imputed"] = False
    for idx in gaps:                       # loop over each gap and fill it
        df.loc[idx, "Waiting_Time_Min"] = group_median[idx]
        df.loc[idx, "Wait_Imputed"] = True
    print(f"Filled {len(gaps)} gaps using Department + Type median.")
    print(f"Remaining blanks (not applicable)  : {df['Waiting_Time_Min'].isnull().sum()}")
    return df


def check_duplicates(df):
    """Task 126 - Check duplicate Appointment_ID values and remove them."""
    heading("TASK 126 - DUPLICATE APPOINTMENT_ID CHECK")
    dup_count = df["Appointment_ID"].duplicated().sum()
    print(f"Duplicate Appointment_ID rows: {dup_count}")
    if dup_count > 0:
        print("\nDuplicated IDs:")
        print(sorted(df.loc[df["Appointment_ID"].duplicated(keep=False), "Appointment_ID"].unique()))
        df = df.drop_duplicates(subset="Appointment_ID", keep="first").reset_index(drop=True)
        print(f"\nDuplicates removed (kept first). Rows now: {len(df)}")
    else:
        print("No duplicates found.")
    return df


def prepare_data():
    """Load -> remove duplicates -> fill missing waits -> add helper columns."""
    df = load_data()
    df = df.drop_duplicates(subset="Appointment_ID", keep="first").reset_index(drop=True)
    df["Appointment_Date"] = pd.to_datetime(df["Appointment_Date"])
    df["Weekday"] = df["Appointment_Date"].dt.day_name()
    df["Month"] = df["Appointment_Date"].dt.strftime("%Y-%m")
    group_median = df.groupby(["Department", "Appointment_Type"])["Waiting_Time_Min"].transform("median")
    mask = df["Waiting_Time_Min"].isnull() & (df["Attendance_Status"] == "Attended")
    df.loc[mask, "Waiting_Time_Min"] = group_median[mask]
    df["Wait_Category"] = categorize_wait(df["Waiting_Time_Min"])
    df["Age_Group"] = pd.cut(df["Patient_Age"], bins=[-1, 17, 59, 200],
                             labels=["Child (0-17)", "Adult (18-59)", "Senior (60+)"])
    return df


# ----------------------------------------------------------------------------
# Tasks 127-133 : summaries
# ----------------------------------------------------------------------------
def overall_summary(df):
    """Task 127 - total appointments + attendance-status percentages."""
    heading("TASK 127 - TOTAL APPOINTMENTS & ATTENDANCE STATUS %")
    print(f"Total appointments: {len(df)}\n")
    counts = df["Attendance_Status"].value_counts()
    pct = df["Attendance_Status"].value_counts(normalize=True) * 100
    out = pd.DataFrame({"Count": counts, "Percent": pct})
    print(out)
    save(out, "attendance_status_summary.csv")
    return out


def dept_summary(df):
    """Tasks 128, 129, 131 - average wait, no-show %, average fee per department."""
    heading("TASKS 128/129/131 - DEPARTMENT SUMMARY")
    g = df.groupby("Department").agg(
        Appointments=("Appointment_ID", "count"),
        Avg_Wait_Min=("Waiting_Time_Min", "mean"),
        Avg_Fee=("Consultation_Fee", "mean"),
    )
    g["NoShow_Count"] = df[df["Attendance_Status"] == "No-Show"].groupby("Department").size()
    g["NoShow_Pct"] = g["NoShow_Count"] / g["Appointments"] * 100
    g = g.fillna(0).sort_values("NoShow_Pct", ascending=False)
    print(g)
    print("\nTask 129 - Departments with the highest no-show %:")
    top = g.head(3)
    rank = 1
    for dept, row in top.iterrows():            # loop to print the ranking
        print(f"  {rank}. {dept:<18} {row['NoShow_Pct']:.2f}%")
        rank += 1
    save(g, "department_summary.csv")
    return g


def sms_comparison(df):
    """Task 130 - no-show rate for SMS reminder Yes vs No (uses crosstab)."""
    heading("TASK 130 - SMS REMINDER vs ATTENDANCE (crosstab)")
    counts = pd.crosstab(df["SMS_Reminder"], df["Attendance_Status"], margins=True)
    pct = pd.crosstab(df["SMS_Reminder"], df["Attendance_Status"], normalize="index") * 100
    print("Counts:")
    print(counts)
    print("\nRow percentages:")
    print(pct)
    yes, no = pct.loc["Yes", "No-Show"], pct.loc["No", "No-Show"]
    if yes < no:
        print(f"\nResult: SMS reminders are linked with FEWER no-shows ({yes:.2f}% vs {no:.2f}%).")
    elif yes > no:
        print(f"\nResult: SMS reminders are linked with MORE no-shows ({yes:.2f}% vs {no:.2f}%).")
    else:
        print("\nResult: No difference in no-show rate.")
    save(pct, "sms_vs_attendance_pct.csv")
    return pct


def volume_by_date_dept(df):
    """Task 132 - appointment volume by date and department."""
    heading("TASK 132 - APPOINTMENT VOLUME BY DATE & DEPARTMENT")
    pivot = pd.pivot_table(df, index="Appointment_Date", columns="Department",
                           values="Appointment_ID", aggfunc="count", fill_value=0)
    pivot["Total"] = pivot.sum(axis=1)
    print("Busiest 10 dates:")
    print(pivot.sort_values("Total", ascending=False).head(10))
    print("\nBy weekday:")
    wd = df.groupby("Weekday").size().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
    print(wd)
    print("\nBy month:")
    print(df.groupby("Month").size())
    save(pivot, "volume_by_date_department.csv")
    return pivot


def new_vs_followup(df):
    """Task 133 - waiting times for New vs Follow-up appointments."""
    heading("TASK 133 - WAITING TIME: NEW vs FOLLOW-UP")
    g = df.groupby("Appointment_Type")["Waiting_Time_Min"].agg(["count", "mean", "median", "max"])
    print(g)
    print("\nBy department:")
    print(df.pivot_table(index="Department", columns="Appointment_Type",
                         values="Waiting_Time_Min", aggfunc="mean"))
    save(g, "new_vs_followup_wait.csv")
    return g


# ----------------------------------------------------------------------------
# Tasks 134-137 : NumPy, IQR, categories, fees
# ----------------------------------------------------------------------------
def numpy_stats(df):
    """Task 134 - NumPy median, quartiles, standard deviation of waiting time."""
    heading("TASK 134 - NUMPY STATISTICS (WAITING TIME)")
    waits = df["Waiting_Time_Min"].dropna().to_numpy()
    stats = {
        "Count": len(waits),
        "Mean": np.mean(waits),
        "Median": np.median(waits),
        "Q1 (25th)": np.percentile(waits, 25),
        "Q3 (75th)": np.percentile(waits, 75),
        "Std Dev": np.std(waits, ddof=1),
        "Min": np.min(waits),
        "Max": np.max(waits),
    }
    for k, v in stats.items():
        print(f"{k:<12}: {v:,.2f}")
    return stats


def iqr_outliers(df):
    """Task 135 - unusually long waits using the 1.5 x IQR rule."""
    heading("TASK 135 - UNUSUALLY LONG WAITS (IQR METHOD)")
    waits = df["Waiting_Time_Min"].dropna().to_numpy()
    q1, q3 = np.percentile(waits, [25, 75])
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = max(q1 - 1.5 * iqr, 0)
    print(f"Q1={q1:.1f}  Q3={q3:.1f}  IQR={iqr:.1f}")
    print(f"Normal range: {lower:.1f} to {upper:.1f} minutes")
    out = df[df["Waiting_Time_Min"] > upper].sort_values("Waiting_Time_Min", ascending=False)
    print(f"\nUnusually long waits: {len(out)} appointments ({len(out)/len(waits)*100:.2f}%)")
    if len(out) > 0:
        print(out[["Appointment_ID", "Department", "Doctor_Name", "Appointment_Date",
                   "Appointment_Time", "Waiting_Time_Min"]].head(15))
        print("\nOutliers by department:")
        print(out["Department"].value_counts())
        save(out, "waiting_time_outliers.csv")
    return out, upper


def categorize_wait(series):
    """Task 136 - waiting-time bands (uses if/elif logic via a helper)."""
    def band(x):
        if pd.isnull(x):
            return "Not Applicable"
        elif x <= 15:
            return "Short (0-15)"
        elif x <= 30:
            return "Moderate (16-30)"
        elif x <= 60:
            return "Long (31-60)"
        else:
            return "Very Long (60+)"
    return series.apply(band)


def wait_bands(df):
    heading("TASK 136 - WAITING-TIME CATEGORIES")
    order = ["Short (0-15)", "Moderate (16-30)", "Long (31-60)", "Very Long (60+)"]
    counts = df["Wait_Category"].value_counts().reindex(order, fill_value=0)
    out = pd.DataFrame({"Count": counts, "Percent": counts / counts.sum() * 100})
    print(out)
    print("\nBand share by department (%):")
    ct = pd.crosstab(df["Department"], df["Wait_Category"], normalize="index")[order] * 100
    print(ct)
    save(out, "waiting_time_bands.csv")
    return out


def fee_summary(df):
    """Task 137 - estimated total consultation fees for ATTENDED appointments."""
    heading("TASK 137 - ESTIMATED CONSULTATION FEES (ATTENDED ONLY)")
    att = df[df["Attendance_Status"] == "Attended"]
    total = att["Consultation_Fee"].sum()
    print(f"Attended appointments : {len(att)}")
    print(f"Estimated total fees  : Rs. {total:,.0f}")
    print(f"Average fee per visit : Rs. {att['Consultation_Fee'].mean():,.2f}")
    by_dept = att.groupby("Department")["Consultation_Fee"].agg(["count", "sum", "mean"])
    by_dept.columns = ["Attended", "Total_Fees", "Avg_Fee"]
    by_dept["Share_Pct"] = by_dept["Total_Fees"] / total * 100
    print("\nBy department:")
    print(by_dept.sort_values("Total_Fees", ascending=False))
    lost = df[df["Attendance_Status"] == "No-Show"]["Consultation_Fee"].sum()
    print(f"\nPotential revenue lost to No-Shows (indicative): Rs. {lost:,.0f}")
    save(by_dept, "estimated_fee_summary.csv")
    return by_dept


# ----------------------------------------------------------------------------
# Task 138 : data-driven recommendations (if / elif / else)
# ----------------------------------------------------------------------------
def recommendations(df):
    heading("TASK 138 - FIVE OPERATIONAL RECOMMENDATIONS")
    recs = []

    # 1. No-show vs SMS
    pct = pd.crosstab(df["SMS_Reminder"], df["Attendance_Status"], normalize="index") * 100
    yes, no = pct.loc["Yes", "No-Show"], pct.loc["No", "No-Show"]
    if no > yes:
        recs.append(f"Extend SMS reminders to all appointments: no-show is {no:.1f}% without SMS vs "
                    f"{yes:.1f}% with SMS.")
    else:
        recs.append("SMS reminders show no clear benefit; test other reminder channels (call/WhatsApp).")

    # 2. Worst no-show department
    ns = (df[df["Attendance_Status"] == "No-Show"].groupby("Department").size()
          / df.groupby("Department").size() * 100).sort_values(ascending=False)
    worst = ns.index[0]
    if ns.iloc[0] > 15:
        recs.append(f"Introduce confirmation calls and a waitlist/overbooking buffer in {worst} "
                    f"(highest no-show rate: {ns.iloc[0]:.1f}%).")
    else:
        recs.append(f"Monitor {worst} monthly; no-show ({ns.iloc[0]:.1f}%) is within an acceptable range.")

    # 3. Longest-wait department
    aw = df.groupby("Department")["Waiting_Time_Min"].mean().sort_values(ascending=False)
    if aw.iloc[0] > 30:
        recs.append(f"Review staffing and slot length in {aw.index[0]} (average wait {aw.iloc[0]:.1f} min); "
                    f"stagger arrival times to cut queues.")
    else:
        recs.append(f"Waits are acceptable overall; keep {aw.index[0]} under watch ({aw.iloc[0]:.1f} min).")

    # 4. Peak day / hour
    busiest_day = df["Weekday"].value_counts().idxmax()
    df_h = df.assign(Hour=df["Appointment_Time"].str[:2].astype(int))
    peak_hour = df_h.groupby("Hour")["Waiting_Time_Min"].mean().idxmax()
    recs.append(f"Add extra staff on {busiest_day} (busiest day) and spread bookings away from "
                f"{peak_hour}:00, which has the longest average wait.")

    # 5. Outliers & new vs follow-up
    waits = df["Waiting_Time_Min"].dropna().to_numpy()
    q1, q3 = np.percentile(waits, [25, 75])
    upper = q3 + 1.5 * (q3 - q1)
    n_out = int((waits > upper).sum())
    t = df.groupby("Appointment_Type")["Waiting_Time_Min"].mean()
    recs.append(f"Set an alert for waits above {upper:.0f} min ({n_out} cases found) and reserve "
                f"dedicated slots for New patients (avg {t['New']:.1f} min vs Follow-up {t['Follow-up']:.1f} min).")

    for i, r in enumerate(recs, start=1):
        print(f"{i}. {r}")
    save(pd.Series(recs, name="Recommendation").to_frame(), "recommendations.csv")
    return recs


# ----------------------------------------------------------------------------
# CUSTOM FILTER & SORT  (user chooses the parameters)
# ----------------------------------------------------------------------------
SORT_COLUMNS = {
    "1": ("Appointment_Date", "Date"),
    "2": ("Waiting_Time_Min", "Waiting time"),
    "3": ("Consultation_Fee", "Consultation fee"),
    "4": ("Patient_Age", "Patient age"),
    "5": ("Department", "Department (A-Z)"),
    "6": ("Doctor_Name", "Doctor (A-Z)"),
    "7": ("Appointment_ID", "Appointment ID"),
}


def ask_choice(df, column, label):
    """Show the unique values of a column; user picks by number or name (comma-separated). Enter = skip."""
    options = sorted(df[column].dropna().astype(str).unique())
    print(f"\n{label}")
    for i, opt in enumerate(options, start=1):
        print(f"   {i}. {opt}")
    while True:
        raw = input("   Enter number(s) or name(s) separated by comma (Enter = ALL): ").strip()
        if raw == "":
            return None
        chosen, valid = [], True
        for token in raw.split(","):
            token = token.strip()
            if token.isdigit() and 1 <= int(token) <= len(options):
                chosen.append(options[int(token) - 1])
            elif token.lower() in [o.lower() for o in options]:
                chosen.append([o for o in options if o.lower() == token.lower()][0])
            else:
                print(f"   '{token}' is not valid. Try again.")
                valid = False
                break
        if valid:
            return chosen


def ask_number(prompt):
    """Ask for a number; Enter = skip (returns None)."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            print("   Please enter a valid number.")


def ask_date(prompt):
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        try:
            return pd.to_datetime(raw, format="%Y-%m-%d")
        except ValueError:
            print("   Use format YYYY-MM-DD (example 2026-02-15).")


def custom_filter(df):
    heading("CUSTOM FILTER & SORT - YOU DECIDE THE PARAMETERS")
    print("For every question below, press Enter to SKIP that filter.")
    result = df.copy()
    applied = []

    # --- categorical filters (loop over a list of (column, label)) ---
    cat_filters = [
        ("Department", "1) DEPARTMENT"),
        ("Doctor_Name", "2) DOCTOR"),
        ("Appointment_Type", "3) APPOINTMENT TYPE"),
        ("Attendance_Status", "4) ATTENDANCE STATUS"),
        ("SMS_Reminder", "5) SMS REMINDER SENT?"),
        ("Gender", "6) GENDER"),
        ("Age_Group", "7) AGE GROUP"),
        ("Weekday", "8) WEEKDAY"),
        ("Wait_Category", "9) WAITING-TIME CATEGORY"),
    ]
    for col, label in cat_filters:
        picked = ask_choice(result, col, label)
        if picked is not None:
            result = result[result[col].astype(str).isin(picked)]
            applied.append(f"{col} in {picked}")

    # --- numeric ranges ---
    print("\n10) WAITING TIME RANGE (minutes)")
    lo = ask_number("   Minimum wait (Enter = skip): ")
    hi = ask_number("   Maximum wait (Enter = skip): ")
    if lo is not None:
        result = result[result["Waiting_Time_Min"] >= lo]
        applied.append(f"Wait >= {lo:g}")
    if hi is not None:
        result = result[result["Waiting_Time_Min"] <= hi]
        applied.append(f"Wait <= {hi:g}")

    print("\n11) CONSULTATION FEE RANGE (Rs.)")
    lo = ask_number("   Minimum fee (Enter = skip): ")
    hi = ask_number("   Maximum fee (Enter = skip): ")
    if lo is not None:
        result = result[result["Consultation_Fee"] >= lo]
        applied.append(f"Fee >= {lo:g}")
    if hi is not None:
        result = result[result["Consultation_Fee"] <= hi]
        applied.append(f"Fee <= {hi:g}")

    print("\n12) PATIENT AGE RANGE")
    lo = ask_number("   Minimum age (Enter = skip): ")
    hi = ask_number("   Maximum age (Enter = skip): ")
    if lo is not None:
        result = result[result["Patient_Age"] >= lo]
        applied.append(f"Age >= {lo:g}")
    if hi is not None:
        result = result[result["Patient_Age"] <= hi]
        applied.append(f"Age <= {hi:g}")

    print("\n13) DATE RANGE (data covers 2026-01-01 to 2026-03-31)")
    d1 = ask_date("   From date YYYY-MM-DD (Enter = skip): ")
    d2 = ask_date("   To date   YYYY-MM-DD (Enter = skip): ")
    if d1 is not None:
        result = result[result["Appointment_Date"] >= d1]
        applied.append(f"Date >= {d1.date()}")
    if d2 is not None:
        result = result[result["Appointment_Date"] <= d2]
        applied.append(f"Date <= {d2.date()}")

    # --- sorting ---
    print("\n14) SORT RESULTS BY")
    for k, (_, nice) in SORT_COLUMNS.items():
        print(f"   {k}. {nice}")
    sort_choice = input("   Choose 1-7 (Enter = no sorting): ").strip()
    if sort_choice in SORT_COLUMNS:
        col = SORT_COLUMNS[sort_choice][0]
        order = input("   Order - A for ascending / D for descending [A]: ").strip().upper()
        if order == "D":
            result = result.sort_values(col, ascending=False)
        else:
            result = result.sort_values(col, ascending=True)
        applied.append(f"Sorted by {col} ({'desc' if order == 'D' else 'asc'})")
    elif sort_choice != "":
        print("   Invalid choice - no sorting applied.")

    # --- how many rows to show ---
    top_n = ask_number("\n15) How many rows to display? (Enter = 20): ")
    top_n = 20 if top_n is None else int(top_n)

    # --- output ---
    heading("RESULT")
    print("Filters applied:", "; ".join(applied) if applied else "None (all records)")
    print(f"Matching records: {len(result)} of {len(df)}")
    if len(result) == 0:
        print("\nNo records match these conditions. Try relaxing a filter.")
        return result

    show_cols = ["Appointment_ID", "Department", "Doctor_Name", "Appointment_Date", "Appointment_Time",
                 "Appointment_Type", "SMS_Reminder", "Attendance_Status", "Waiting_Time_Min",
                 "Consultation_Fee"]
    print(result[show_cols].head(top_n).to_string(index=False))

    print("\nQuick stats for the filtered data:")
    print(f"  Average waiting time : {result['Waiting_Time_Min'].mean():.2f} min")
    print(f"  Median waiting time  : {result['Waiting_Time_Min'].median():.2f} min")
    print(f"  Average fee          : Rs. {result['Consultation_Fee'].mean():,.2f}")
    print("  Attendance split (%):")
    for status, pct in (result["Attendance_Status"].value_counts(normalize=True) * 100).items():
        print(f"     {status:<10} {pct:.1f}%")

    if input("\nSave this filtered data to CSV? (y/n): ").strip().lower() == "y":
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "custom_filtered_data.csv")
        result.to_csv(path, index=False)
        print(f"[saved] {path}")
    return result


# ----------------------------------------------------------------------------
# MENU
# ----------------------------------------------------------------------------
def run_all(df_raw, df):
    inspect_data(df_raw)
    handle_missing_waiting(df_raw.copy())
    check_duplicates(df_raw.copy())
    overall_summary(df)
    dept_summary(df)
    sms_comparison(df)
    volume_by_date_dept(df)
    new_vs_followup(df)
    numpy_stats(df)
    iqr_outliers(df)
    wait_bands(df)
    fee_summary(df)
    recommendations(df)


MENU = """
================ HOSPITAL APPOINTMENT OPERATIONS ANALYSIS ================
  1. Load & inspect data                    (Task 124)
  2. Missing waiting times                  (Task 125)
  3. Duplicate Appointment_ID check         (Task 126)
  4. Total appointments & attendance %      (Task 127)
  5. Department summary: wait/no-show/fee   (Tasks 128, 129, 131)
  6. SMS reminder vs no-show                (Task 130)
  7. Volume by date & department            (Task 132)
  8. New vs Follow-up waiting time          (Task 133)
  9. NumPy statistics of waiting time       (Task 134)
 10. Unusually long waits (IQR)             (Task 135)
 11. Waiting-time categories                (Task 136)
 12. Estimated fees (attended only)         (Task 137)
 13. Five operational recommendations       (Task 138)
 14. CUSTOM FILTER & SORT (choose your own parameters)
 15. Run ALL tasks and save reports
  0. Exit
==========================================================================="""


def main():
    global INTERACTIVE
    INTERACTIVE = "--all" not in sys.argv
    df_raw = load_data()
    df = prepare_data()

    if not INTERACTIVE:
        run_all(df_raw, df)
        print("\nAll tasks complete. Reports saved in the 'output' folder.")
        return

    actions = {
        "1": lambda: inspect_data(df_raw),
        "2": lambda: handle_missing_waiting(df_raw.copy()),
        "3": lambda: check_duplicates(df_raw.copy()),
        "4": lambda: overall_summary(df),
        "5": lambda: dept_summary(df),
        "6": lambda: sms_comparison(df),
        "7": lambda: volume_by_date_dept(df),
        "8": lambda: new_vs_followup(df),
        "9": lambda: numpy_stats(df),
        "10": lambda: iqr_outliers(df),
        "11": lambda: wait_bands(df),
        "12": lambda: fee_summary(df),
        "13": lambda: recommendations(df),
        "14": lambda: custom_filter(df),
        "15": lambda: run_all(df_raw, df),
    }
    while True:
        print(MENU)
        choice = input("Enter your choice (0-15): ").strip()
        if choice == "0":
            print("Goodbye!")
            break
        elif choice in actions:
            actions[choice]()
            pause()
        else:
            print("Invalid choice. Please enter a number from 0 to 15.")


INTERACTIVE = True

if __name__ == "__main__":
    main()
