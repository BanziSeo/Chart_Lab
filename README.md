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
You can also use the provided helper scripts which set up a virtual
environment, install dependencies and then start the app:

- Windows: `run_app.bat`
- Linux/macOS: `./run_app.sh`

## Helper scripts
- **run_app.bat** – Creates a local virtual environment under `venv`, installs
  dependencies from `Chart_Lab/requirements.txt` and runs the Streamlit
  application.
- **run_app.sh** – Same as `run_app.bat` but for Unix-like systems.
- **update_repo.bat** – Simplifies committing and pushing changes. It stages all
  files, asks for a commit message, pulls the latest `main` with rebase and then
  pushes your commits.
