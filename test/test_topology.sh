#!/bin/bash
if test ! -e /var/www/exo/python/bin/python3; then
    /usr/bin/python3 -m venv --system-site-packages /var/www/exo/python/
fi
/var/www/exo/python/bin/pip install pyyaml
/var/www/exo/python/bin/python3 /var/www/exo/validate_topology.py
