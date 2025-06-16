# Chart Lab

A small Streamlit application for chart-based trading practice.

## Setup
1. Install the required packages:
   ```bash
   pip install -r Chart_Lab/requirements.txt
   ```

## Running the app
Launch the app with Streamlit:
```bash
streamlit run Chart_Lab/app.py
```
You can also run the `run_app.bat` script on Windows which sets up a virtual
environment, installs dependencies and starts the app for you.

## Modelbook for Random Games
The "모델북 랜덤 시작" button loads tickers from a local text file when picking a
random game. The expected file is `Chart_Lab/modelbook.txt` and it should
contain ticker symbols separated by commas, e.g.

```text
MSFT,AAPL,GOOG
```

You can edit this file or replace it entirely with your own list of tickers. The
application will shuffle the entries and start a game using the first ticker for
which data is available.

## Helper scripts
- **run_app.bat** – Creates a local virtual environment under `venv`, installs
  dependencies from `Chart_Lab/requirements.txt` and runs the Streamlit
  application.
- **update_repo.bat** – Simplifies committing and pushing changes. It stages all
  files, asks for a commit message, pulls the latest `main` with rebase and then
  pushes your commits.
