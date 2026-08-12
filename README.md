# 🗳️ India Election Analysis Dashboard

An interactive Streamlit dashboard for analyzing Indian Lok Sabha election data from **1984 to 2024**.

---

## 📁 Project Structure

```
├── app.py                   ← Main Streamlit application
├── requirements.txt         ← Python dependencies
├── elections_overview.csv   ← National-level election summary
├── party_results.csv        ← Party-wise seats & vote share
├── state_results.csv        ← State-wise seat breakdown
├── constituency_data.csv    ← Constituency-level results
├── demographics_data.csv    ← Women, youth, first-time MPs
├── alliance_data.csv        ← Alliance (NDA/UPA/INDIA) totals
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run india_election_dashboard_final.py
```

The app will open at `http://localhost:8501`

---

## 📊 Dashboard Tabs

| Tab | What it shows |
|-----|---------------|
| 📊 Overview | Seat bars, vote share pie, efficiency scatter |
| 🏛️ Party Performance | Trends, area charts, win strike rates |
| 🗺️ State Analysis | Dominant party by state, drill-downs, history |
| 📈 Trends & Turnout | Turnout trend, voter growth, PM timeline |
| ⚔️ Head-to-Head | Any two-party comparison with scorecard |
| 🤝 Alliance Battle | NDA vs UPA vs INDIA coalition analysis |
| 🏟️ Constituencies | Winning margins, voter counts per seat |
| 👥 Demographics | Women MPs, first-time MPs, youth MPs |
| 🔮 Insights | Auto-generated key insights + heatmap |

---

## 📂 CSV File Guide

### `elections_overview.csv`
National summary per election year — turnout, registered voters, winning party, PM.

### `party_results.csv`
Every major party's seats won, vote share, seats contested, and alliance per year.

### `state_results.csv`
State-wise seat count broken down by party for 2009–2024. Also includes turnout and winning party.

### `constituency_data.csv`
Individual constituency results including winner, runner-up, margin, and total votes.

### `demographics_data.csv`
Women candidates, first-time MPs, youth MPs, and incumbent re-elections per year.

### `alliance_data.csv`
Alliance-level totals (NDA, UPA, INDIA etc.) per election year.

---

## 🛠️ Customisation Tips

- **Add more states**: Extend `state_results.csv` with more rows
- **Add more years**: Extend any CSV with earlier years (1984, 1989, 1991...)
- **Add constituency data**: More rows in `constituency_data.csv`
- **Change theme**: Modify the CSS in the `st.markdown("""...""")` block in `app.py`

---

## 📜 Data Source
Election Commission of India — [https://eci.gov.in](https://eci.gov.in)
