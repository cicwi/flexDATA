# -*- coding: utf-8 -*-
"""
An example of SCP transfer.
"""
#%% Imports:

from flexdata import scp
from flexdata import io

#%% Data copy:
local = '/ufs/ciacc/flexbox/test1'
remote = '/ufs/ciacc/flexbox/test2'
host = 'localhost'
user = 'anonymous'

# Get files from remote:
scp.ssh_get_path(local, remote, host, user)

# Send files to remote location:
scp.ssh_put_path(local, remote, host, user)

# Remove data (use with care!!!):
io.delete_path(local)