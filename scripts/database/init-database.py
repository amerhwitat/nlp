#!/usr/bin/env python3
import argparse, os, shutil, sqlite3, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['check','init']); ap.add_argument('--sqlite',default=str(ROOT/'artifacts/database/nlp.sqlite')); args=ap.parse_args()
    if args.mode=='check':
        print('sqlite3:', shutil.which('sqlite3') or 'python sqlite3 module')
        print('psql:', shutil.which('psql') or 'not found')
        return
    sql_dir=ROOT/'db/sql'; Path(args.sqlite).parent.mkdir(parents=True,exist_ok=True)
    if os.getenv('NLP_DATABASE_URL') or os.getenv('DATABASE_URL'):
        url=os.getenv('NLP_DATABASE_URL') or os.getenv('DATABASE_URL')
        if not shutil.which('psql'): raise SystemExit('DATABASE_URL is set but psql is unavailable')
        for f in sorted(sql_dir.glob('*.sql')): subprocess.run(['psql',url,'-v','ON_ERROR_STOP=1','-f',str(f)],check=True)
    else:
        con=sqlite3.connect(args.sqlite)
        try:
            for f in sorted(sql_dir.glob('*.sql')): con.executescript(f.read_text(encoding='utf-8'))
            con.commit()
        finally: con.close()
if __name__=='__main__': main()
