#!/usr/bin/env python3
from tinygrad.helpers import db_connection, VERSION, getenv
cur = db_connection()
cur.execute(f"drop table if exists process_replay_{getenv('GITHUB_RUN_ID', 'HEAD')}_{VERSION}")
