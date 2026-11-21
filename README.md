# Project 9: Hospital Appointment Operations Analysis

Operational (non-clinical) analysis of hospital appointments using **Python, pandas, NumPy, if/else and loops**.
No medical conclusions are drawn from this data.

## Files
| File | Purpose |
|---|---|
| `09_hospital_appointments.csv` | Dataset: 1,612 rows (1,600 unique + 12 duplicate IDs), 13 columns |
| `generate_dataset.py` | Re-creates the dataset (seeded, so results are reproducible) |
| `hospital_analysis.py` | Main menu-driven program (Tasks 124-138 + custom filter/sort) |
| `SRS_Hospital_Appointment_Analysis.docx` | Software Requirements Specification |
| `output/` | Sample CSV reports created by the program |

## How to run
```bash
pip install -r requirements.txt
python hospital_analysis.py          # interactive menu
python hospital_analysis.py --all    # runs every task and saves reports to output/
```

## Task map
| Task | Menu option |
|---|---|
| 124 Load & inspect | 1 |
| 125 Missing waiting times | 2 |
| 126 Duplicate Appointment_ID | 3 |
| 127 Totals & attendance % | 4 |
| 128, 129, 131 Dept wait / no-show / fee | 5 |
| 130 SMS vs no-show (crosstab) | 6 |
| 132 Volume by date & dept | 7 |
| 133 New vs Follow-up wait | 8 |
| 134 NumPy median/quartiles/std | 9 |
| 135 IQR outliers | 10 |
| 136 Wait categories | 11 |
| 137 Estimated fees (attended) | 12 |
| 138 Five recommendations | 13 |
| Custom filter & sort | 14 |

## Custom filter & sort (option 14)
Press **Enter** to skip any filter. Multiple values can be typed with commas (e.g. `1,3` or `Cardiology,Neurology`).

| # | Parameter | What to enter | Example |
|---|---|---|---|
| 1 | Department | number(s) or name(s) | `Cardiology,Neurology` |
| 2 | Doctor | number(s) or name(s) | `2` |
| 3 | Appointment type | New / Follow-up | `New` |
| 4 | Attendance status | Attended / Cancelled / No-Show | `No-Show` |
| 5 | SMS reminder | Yes / No | `No` |
| 6 | Gender | Female / Male | `Female` |
| 7 | Age group | Child (0-17) / Adult (18-59) / Senior (60+) | `Senior (60+)` |
| 8 | Weekday | Monday ... Saturday | `Monday` |
| 9 | Wait category | Short / Moderate / Long / Very Long | `Very Long (60+)` |
| 10 | Waiting time range | min and/or max minutes | `45` to `120` |
| 11 | Fee range | min and/or max Rs. | `500` to `1000` |
| 12 | Age range | min and/or max | `60` to `90` |
| 13 | Date range | `YYYY-MM-DD` (2026-01-01 to 2026-03-31) | `2026-02-01` to `2026-02-28` |
| 14 | Sort by | 1-7, then A (ascending) / D (descending) | `2` then `D` |
| 15 | Rows to show | number | `10` |

The result shows matching records, quick statistics, and offers to save a CSV to `output/custom_filtered_data.csv`.

## Upload to GitHub
```bash
git init
git add .
git commit -m "Hospital Appointment Operations Analysis"
git branch -M main
git remote add origin https://github.com/<your-username>/hospital-appointment-analysis.git
git push -u origin main
```
Submit the SRS document, `hospital_analysis.py`, and the repository link on the LMS.
