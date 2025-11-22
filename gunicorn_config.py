import os
import multiprocessing

# Bind to the PORT environment variable (Render sets this)
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Number of worker processes
workers = multiprocessing.cpu_count() * 2 + 1

# Worker class
worker_class = 'sync'

# Timeout for workers
timeout = 120

# Access log
accesslog = '-'

# Error log
errorlog = '-'

# Log level
loglevel = 'info'

# Preload app
preload_app = True
