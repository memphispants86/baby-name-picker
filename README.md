# Baby A Name Picker

A Streamlit application for rating and ranking baby names with regional popularity data from NSW, VIC, UK, and USA.

## Features

- **Name Rating**: Rate names on a scale of 0-10 for each spouse
- **Weighting Methods**: Choose between Percentile and Z-score normalization
- **Regional Data**: View name popularity across NSW, VIC, UK, and USA
- **Expected Births**: Calculate expected "X Martin" births per year
- **Data Management**: Add new names and edit existing ratings

## Quick Start

### Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the data normalization script:
   ```bash
   python normalize_rankings.py
   ```
5. Start the app:
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Connect your GitHub account
5. Select this repository
6. Set the main file path to `app.py`
7. Click "Deploy!"

## Data Sources

The app uses pre-normalized data from:
- **NSW**: `popular-baby-names-1952-to-2024.csv`
- **VIC**: Excel files from Victoria directory
- **UK**: `UK_all_years_boys_EW.csv`
- **USA**: `yob*.txt` and `yob*.csv` files

## File Structure

```
├── app.py                    # Main Streamlit application
├── normalize_rankings.py     # Data normalization script
├── requirements.txt          # Python dependencies
├── normalized_rankings.json # Pre-processed regional data
├── names.db                  # SQLite database (created automatically)
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── README.md                # This file
```

## Configuration

The app uses SQLite for local data storage and includes:
- Spouse management
- Name ratings and rankings
- Regional popularity data
- Expected birth calculations

## Troubleshooting

If the app crashes or runs slowly:
1. Check the log file: `tail -f app.log`
2. Run with console logging: `LOG_TO_STDOUT=1 streamlit run app.py`
3. Ensure `normalized_rankings.json` exists (run `normalize_rankings.py`)

## License

MIT License - feel free to use and modify as needed.