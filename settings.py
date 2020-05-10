from dotenv import load_dotenv
load_dotenv()

import os
API_KEY = os.getenv("API_KEY")
CSE_ID = os.getenv("SEARCH_ENGINE_ID")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

crawled_data_dir = os.path.join(PROJECT_ROOT, 'data_crawled')