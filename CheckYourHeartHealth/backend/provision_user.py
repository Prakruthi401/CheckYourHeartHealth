"""Local operator CLI. Never expose this as a public signup route."""
import argparse
from getpass import getpass
from src.database import init_database, add_user

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provision a doctor or admin from the trusted deployment console.')
    parser.add_argument('username')
    parser.add_argument('--role', choices=['doctor','admin'], required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    password = getpass('Unique password (12+ characters): ')
    if password != getpass('Confirm: '):
        raise SystemExit('Passwords differ')
    init_database()
    add_user(args.username, password, args.name, args.role)
    print('Account created. No password was logged.')
