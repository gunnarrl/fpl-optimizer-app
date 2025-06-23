This document provides a granular, step-by-step plan for building the FPL Optimizer MVP. Each task is designed to be small, testable, and focused on a single concern.

---

### **Phase 1: Project Setup & Initial Configuration**

#### **Task 1: Initialize Project Directory Structure**
* **Goal:** Create the main folders for the monorepo.
* **Instructions:**
    1.  Create the root directory: `fpl-optimizer-app/`
    2.  Inside the root, create the following subdirectories:
        * `backend/`
        * `frontend/`
        * `data_pipeline/`
* **How to Test:** Verify that the `fpl-optimizer-app` directory contains the three specified subdirectories.

#### **Task 2: Set Up Supabase Project**
* **Goal:** Create the Supabase project and the database table to store player predictions.
* **Instructions:**
    1.  Go to [supabase.com](https://supabase.com) and create a new project.
    2.  Navigate to the "SQL Editor" section.
    3.  Run the following SQL query to create the `player_predictions` table. This schema is based on the output of `autoGlueon1.py`.
        ```sql
        CREATE TABLE player_predictions (
            element BIGINT,
            "GW" BIGINT,
            name TEXT,
            value REAL,
            team TEXT,
            position TEXT,
            predicted_points_gw_1 REAL,
            points_gw_1 REAL,
            predicted_points_gw_2 REAL,
            points_gw_2 REAL,
            predicted_points_gw_3 REAL,
            points_gw_3 REAL,
            predicted_points_gw_4 REAL,
            points_gw_4 REAL,
            predicted_points_gw_5 REAL,
            points_gw_5 REAL,
            predicted_points_gw_6 REAL,
            points_gw_6 REAL,
            PRIMARY KEY (element, "GW")
        );
        ```
    4.  Go to Project Settings > API and find your **Project URL** and **`service_role` key**.
* **How to Test:** The `player_predictions` table exists in your Supabase project with the correct columns.

#### **Task 3: Set Up Vercel Project**
* **Goal:** Create a Vercel project linked to your Git repository.
* **Instructions:**
    1.  Initialize a Git repository in your `fpl-optimizer-app` root folder.
    2.  Push the repository to GitHub/GitLab/Bitbucket.
    3.  Go to [vercel.com](https://vercel.com) and create a new project, importing the repository you just created.
    4.  In the project settings, set the "Framework Preset" to **Next.js**.
    5.  Set the "Root Directory" to `frontend`.
* **How to Test:** A project is created on Vercel and is linked to your repository. Initial deployments may fail, which is expected.

#### **Task 4: Configure Environment Variables**
* **Goal:** Securely store your Supabase credentials for local and production environments.
* **Instructions:**
    1.  In the `fpl-optimizer-app` root directory, create a file named `.env.local`.
    2.  Add your Supabase credentials to this file:
        ```
        SUPABASE_URL="YOUR_SUPABASE_PROJECT_URL"
        SUPABASE_KEY="YOUR_SUPABASE_SERVICE_ROLE_KEY"
        ```
    3.  In your Vercel project settings, navigate to "Environment Variables" and add `SUPABASE_URL` and `SUPABASE_KEY` with the same values.
* **How to Test:** Your local and Vercel environments have access to the Supabase credentials.

---

### **Phase 2: Data Pipeline (Offline Task)**

**Assumption:** This pipeline will be run in an environment (e.g., your local machine, a dedicated server, or a powerful CI/CD runner) that can handle its dependencies and computational load.

#### **Task 5: Organize Data Pipeline Scripts**
* **Goal:** Move all your existing Python scripts into the `data_pipeline` directory.
* **Instructions:**
    1.  Copy all your `src` subdirectories (`data_ingestion`, `feature_engineering`, `modeling`, etc.) into `data_pipeline/src/`.
* **How to Test:** The `data_pipeline/src` directory now contains all your original Python code.

#### **Task 6: Create `upload_to_supabase.py`**
* **Goal:** Create the Python script to upload prediction results to your database.
* **File to Create:** `data_pipeline/upload_to_supabase.py`
* **Instructions:** Use the code provided in the architecture document. Make sure it reads the `predictions_autogluon.csv` file, which is the output of `autoGlueon1.py`.
* **How to Test:** When run manually (after the model has created the CSV), the script should not produce errors and you should see data appear in your Supabase table.

#### **Task 7: Modify `autoGlueon1.py` for Vercel Compatibility**
* **Goal:** Adjust the modeling script to use relative paths and handle dependencies correctly.
* **File to Modify:** `data_pipeline/src/modeling/autoGlueon1.py`
* **Instructions:**
    1.  Find the line `PREDICTIONS_OUTPUT_FILE = 'predictions_autogluon.csv'` and ensure it saves the file to the root of the `data_pipeline` directory, so `upload_to_supabase.py` can find it. Change it to: `PREDICTIONS_OUTPUT_FILE = '../../predictions_autogluon.csv'`.
    2.  Do the same for `INPUT_DATA_FILE` to ensure it reads from the correct path relative to its own location.
* **How to Test:** The script runs without file path errors and generates the CSV in the `data_pipeline` root.

#### **Task 8: Create the Master Pipeline Script `run_weekly_pipeline.sh`**
* **Goal:** Create a shell script to run the entire data pipeline in the correct order.
* **File to Create:** `data_pipeline/run_weekly_pipeline.sh`
* **Instructions:**
    1.  Use the code from the architecture document.
    2.  Ensure the Python commands are correct for your system (`python` vs. `python3`).
    3.  Make the script executable: `chmod +x run_weekly_pipeline.sh`.
* **How to Test:** Running `./run_weekly_pipeline.sh` from the `data_pipeline` directory executes all scripts in sequence and populates the Supabase database.

---

### **Phase 3: Backend API (Python on Vercel)**

#### **Task 9: Set Up FastAPI Project Structure for Vercel**
* **Goal:** Configure the `backend` directory so Vercel can deploy it as a serverless function.
* **Instructions:**
    1.  Inside `backend/`, create a `requirements.txt` file with the following content:
        ```
        fastapi
        uvicorn
        pulp
        pandas
        supabase-py
        httpx
        ```
    2.  Inside `backend/`, create an `app/` directory.
    3.  In the project root (`fpl-optimizer-app/`), create a `vercel.json` file. This tells Vercel how to handle the monorepo:
        ```json
        {
          "builds": [
            {
              "src": "frontend/next.config.js",
              "use": "@vercel/next"
            },
            {
              "src": "backend/main.py",
              "use": "@vercel/python"
            }
          ],
          "routes": [
            {
              "src": "/api/(.*)",
              "dest": "backend/main.py"
            },
            {
              "src": "/(.*)",
              "dest": "frontend/$1"
            }
          ]
        }
        ```
* **How to Test:** When deployed, Vercel will attempt to build both a Next.js app and a Python API.

#### **Task 10: Create Database and FPL API Clients**
* **Goal:** Create the helper modules for database and external API interactions.
* **Files to Create:** `backend/app/db_client.py` and `backend/app/fpl_api_client.py`.
* **Instructions:**
    1.  Use the code provided in the architecture document for both files.
    2.  In `fpl_api_client.py`, add a new function to get the current gameweek:
        ```python
        def get_current_gameweek():
            """Fetches the current live gameweek ID."""
            with httpx.Client() as client:
                response = client.get(f"{FPL_API_URL}bootstrap-static/")
                response.raise_for_status()
                events = response.json()['events']
                current_gw = next((event['id'] for event in events if event['is_current']), None)
                return current_gw
        ```
* **How to Test:** The functions can be imported and run in a separate Python script without errors.

#### **Task 11: Refactor `fpl-optimization.py` into a Class**
* **Goal:** Convert your core optimization script into a reusable Python class.
* **File to Create:** `backend/app/optimizer.py`
* **Instructions:**
    1.  Copy the entire `FPLOptimizer` class from your `fpl-optimization.py` or `debugging.py` script into this new file.
    2.  Modify the `__init__` method to accept a pandas DataFrame as an argument instead of reading a file: `def __init__(self, predictions_df: pd.DataFrame, config_file=None):`.
    3.  Modify the `load_data` method to accept the DataFrame: `def load_data(self, predictions_df: pd.DataFrame):`. Remove the file-reading part from this method.
    4.  Modify `optimize_single_gw` to take `current_squad`, `bank`, `free_transfers`, and `used_chips` as direct arguments instead of calling `load_current_squad`.
* **How to Test:** The class can be instantiated and its methods called from another Python script without errors.

#### **Task 12: Create the Main FastAPI Endpoint**
* **Goal:** Create the API server that ties all backend components together.
* **File to Create:** `backend/main.py`
* **Instructions:** Use the code from the architecture document. Make sure it correctly imports from the `app` module.
* **How to Test:** Run `uvicorn backend.main:app --reload` from the project root. Send a POST request to `http://127.0.0.1:8000/api/optimize` with a valid JSON body (e.g., `{"manager_id": 123, "gameweek": 36}`). You should get back a JSON response or a specific error.

---

### **Phase 4: Frontend (Next.js)**

#### **Task 13: Initialize Next.js App**
* **Goal:** Create a boilerplate Next.js application.
* **Instructions:** Run `npx create-next-app@latest frontend` from the project root. This will set up all necessary files inside the `frontend` directory.
* **How to Test:** Running `npm run dev` inside the `frontend` directory starts the development server successfully.

#### **Task 14: Create Homepage UI**
* **Goal:** Build the form for user input.
* **File to Modify:** `frontend/pages/index.js`
* **Instructions:** Use the JSX code from the architecture document to create the input form for the FPL Manager ID.
* **How to Test:** The homepage displays the input field and button. Entering an ID and clicking the button navigates to the (currently non-existent) results page.

#### **Task 15: Create Results Page and API Fetching Logic**
* **Goal:** Set up the dynamic results page that calls the backend API.
* **File to Create:** `frontend/pages/results/[managerId].js`
* **Instructions:**
    1.  Install SWR: `npm install swr` inside the `frontend` directory.
    2.  Use the code from the architecture document.
    3.  Modify the `fetcher` function to point to your backend API route: `/api/optimize`.
    4.  Replace the hardcoded `currentGameweek` by calling the new `get_current_gameweek` endpoint you will create in the backend. (For now, keeping it hardcoded is fine for testing).
* **How to Test:** Navigating to `/results/123` shows the "Optimizing your team..." message, then either displays an error or the raw JSON data from the API.

#### **Task 16: Build Basic Display Components**
* **Goal:** Create placeholder components for the UI.
* **Files to Create:**
    * `frontend/components/Pitch.js`
    * `frontend/components/PlayerCard.js`
    * `frontend/components/Transfers.js`
* **Instructions:** Create basic functional components that receive props and render simple text. For example:
    ```jsx
    // frontend/components/PlayerCard.js
    export default function PlayerCard({ player }) {
      return <div>{player.name} ({player.cost / 10}M)</div>;
    }
    ```
* **How to Test:** Import and use these components in `[managerId].js`. The page should render the placeholder content.

#### **Task 17: Populate UI Components with Data**
* **Goal:** Make the components display the actual data from the API response.
* **Files to Modify:** `Pitch.js`, `PlayerCard.js`, `Transfers.js`, and `[managerId].js`.
* **Instructions:**
    1.  In `Transfers.js`, map over the `transfersIn` and `transfersOut` props to display tables of player transfers.
    2.  In `Pitch.js`, map over the `startingXI` and `bench` props, rendering a `PlayerCard` for each player. Use CSS Flexbox or Grid to arrange them in a pitch-like formation.
* **How to Test:** The results page now correctly displays the optimized team and transfers in a structured format.

---

### **Phase 5: Deployment & Final Polish**

#### **Task 18: Finalize and Deploy**
* **Goal:** Deploy the full-stack application to Vercel.
* **Instructions:**
    1.  Push all your changes to your Git repository.
    2.  Vercel will automatically trigger a new deployment.
    3.  Monitor the build logs in the Vercel dashboard for any errors related to the frontend or backend builds.
* **How to Test:** The deployed Vercel URL works, and you can successfully run an optimization on the live site.
