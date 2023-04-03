import os

APP_ENTRY_POINT = 'app.py'
dir_path = os.path.dirname(__file__)
path = os.path.join(dir_path, APP_ENTRY_POINT)

if __name__ == "__main__":
    os.system('streamlit run "{}"'.format(path))

