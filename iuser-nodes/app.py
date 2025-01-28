#!/usr/bin/env python3
from config import create_app

if __name__ == '__main__':
    server = create_app()
    server.run()
