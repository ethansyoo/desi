# User Manual: Setting Up Your chatDESI Application

Welcome to chatDESI! This guide will walk you through setting up the application on your computer from scratch. No prior coding experience is needed—just follow these steps carefully.

### Part A: Initial Software Installation

Before we get the chatDESI code, we need to install a few essential tools.

1.  **Install Python:**
    * Go to the official Python website: [python.org/downloads/](https://www.python.org/downloads/).
    * Download the latest version of Python (anything above 3.9 is fine).
    * Run the installer. **Important:** On the first screen of the installer, make sure to check the box that says "**Add Python to PATH**". This will make the next steps much easier.

2.  **Install Git:**
    * Git is a tool used to copy and manage code.
    * Go to the Git download page: [git-scm.com/downloads](https://git-scm.com/downloads).
    * Download and install the version for your operating system (Windows, Mac, or Linux). You can accept all the default settings during installation.

3.  **Install a Code Editor (Optional but Recommended):**
    * While you can use Notepad or TextEdit, a code editor makes viewing and editing files much easier.
    * We recommend **Visual Studio Code (VS Code)**. It's free and very user-friendly.
    * Download it here: [code.visualstudio.com](https://code.visualstudio.com/).

### Part B: Setting Up the Project Code

Now, let's get the chatDESI code onto your computer.

1.  **Open the Terminal (or Command Prompt):**
    * **On Windows:** Press the `Windows Key`, type `cmd`, and press Enter.
    * **On Mac:** Press `Command + Spacebar`, type `Terminal`, and press Enter.

2.  **Create a Project Folder:**
    * In the terminal, type the following commands one by one, pressing Enter after each:
        ```bash
        mkdir chatDESI_project
        cd chatDESI_project
        ```
    * This creates a new folder named `chatDESI_project` and navigates into it.

3.  **Clone the Code from Git:**
    * Copy the code from the repository by running this command in your terminal:
        ```bash
        git clone [https://github.com/ethansyoo/desi.git](https://github.com/ethansyoo/desi.git) .
        ```
    * *(Note the `.` at the end—it tells Git to copy the files directly into your current folder.)*

4.  **Create a Python Virtual Environment:**
    * A virtual environment is like a private workspace for our project's code, so it doesn't interfere with other Python projects.
    * Run this command:
        ```bash
        python3 -m venv chatdesi_venv
        ```
    * This creates a folder named `chatdesi_venv`.

5.  **Activate the Virtual Environment:**
    * **On Windows:**
        ```bash
        chatdesi_venv\Scripts\activate
        ```
    * **On Mac/Linux:**
        ```bash
        source chatdesi_venv/bin/activate
        ```
    * You'll know it worked if you see `(chatdesi_venv)` at the beginning of your terminal prompt.

6.  **Install Required Packages:**
    * The project uses a number of Python libraries. You can install them all at once by running:
        ```bash
        pip install streamlit pymongo pandas numpy torch transformers scikit-learn PyMuPDF certifi
        ```

### Part C: Setting Up MongoDB Atlas (Your Database)

The application needs a database to store the PDF documents. We'll use MongoDB Atlas, a free cloud database.

1.  **Create a Free MongoDB Atlas Account:**
    * Go to [mongodb.com/cloud/atlas/register](https://www.mongodb.com/cloud/atlas/register) and sign up.

2.  **Create a New Cluster:**
    * Follow the on-screen instructions to create a new cluster. The free **"M0 Sandbox"** tier is perfect for this.
    * Choose a cloud provider and region that is close to you.

3.  **Create a Database User:**
    * In the "Database Access" section, create a new user. Give it a **username** and a **password**. *Save these—you will need them soon.*

4.  **Allow Network Access:**
    * In the "Network Access" section, add your IP address. The easiest way is to click "**Add Current IP Address**."
    * For testing, you can also allow access from anywhere by entering `0.0.0.0/0`, but this is less secure.

5.  **Get Your Connection String:**
    * Go back to your cluster's "Overview" page and click "**Connect**."
    * Choose "**Connect your application**."
    * Copy the connection string provided. It will look something like this:
        `mongodb+srv://<username>:<password>@cluster-name.mongodb.net/?retryWrites=true&w=majority`

### Part D: Final Configuration (Secrets File)

The application needs to know your database connection string and other credentials. We'll store these in a "secrets" file that is kept private.

1.  **Create the Secrets Folder and File:**
    * Inside your `chatDESI_project` folder, create a new folder named `.streamlit`.
    * Inside that new folder, create a file named `secrets.toml`.

2.  **Add Your Credentials to the File:**
    * Open `secrets.toml` with your code editor and paste the following content into it. Replace the placeholder values with your actual credentials.

    ```toml
    # MongoDB Connection Strings
    mongo_general_connection_string = "mongodb+srv://YOUR_GENERAL_USERNAME:YOUR_GENERAL_PASSWORD@your_cluster_url.mongodb.net/"
    mongo_admin_connection_string = "mongodb+srv://YOUR_ADMIN_USERNAME:YOUR_ADMIN_PASSWORD@your_cluster_url.mongodb.net/"

    # Admin password for the application UI
    app_admin_password = "CHOOSE_A_STRONG_PASSWORD"
    ```

    * For both `mongo_general_connection_string` and `mongo_admin_connection_string`, use the string you copied from MongoDB Atlas, but replace `<username>` and `<password>` with the credentials you created.

### Part E: Running the Application

You're all set! Now you can run the app.

1.  **Run the Streamlit Command:**
    * Make sure your virtual environment is still active (`(chatdesi_venv)` should be visible).
    * In your terminal, run:
        ```bash
        streamlit run run_chatdesi.py
        ```

2.  **Open the App:**
    * The command will provide a local URL (usually `http://localhost:8501`). Open this in your web browser.

3.  **Upload Documents (as Admin):**
    * In the sidebar, you'll see an "Admin Login" section. Enter the `app_admin_password` you set in your `secrets.toml` file.
    * Once logged in, the admin panel will appear, allowing you to upload the PDF documents you want the chatbot to use.

Congratulations! Your chatDESI application is now up and running.