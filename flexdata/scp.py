#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kostenko
Created on Oct 2018

This module will contain read / write routines to convert FlexRay scanner data into ASTRA compatible data

We can now read/write:
    image files (tiff stacks)
    log files from Flex ray (settings.txt)
    toml geometry files (metadata.toml)

We can also copy data over SCP!

"""

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Imports >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import os             # operations with filenames
import stat           # same here...
from tqdm import tqdm # progress bar
import paramiko       # SCP client
import errno          # Uesd for error tracking in SCP client
from traceback import print_exception # Error stack printer
from sys import exc_info              # Error info 
import shutil         # Use it to remove files

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
 
def ssh_get_path(local_path, remote_path, hostname, username, password = None, ovewrite = 'different'):
    '''
    Copy files and directories recursively from remote path to local path.
    Set overwrite to: 
        'different' to overwrite files with a different size;
        'always' to overwrite all files
        'never' to never overwrite
    '''
    # Create the destination directory
    if not os.path.exists(local_path):
        os.mkdir(local_path)
        print('Local directory created:', local_path)
    
    # Connect to remote:
    sftp = connect_sftp(hostname, username, password, os.path.join(local_path, 'scp.log'))

    try:
        sftp.get_path(local_path, remote_path, ovewrite)            
        
    except: 
        sftp.close()
        
        info = exc_info()
        print_exception(*info)
        
    finally:        
        sftp.close()
    
def ssh_put_path(local_path, remote_path, hostname, username, password = None, ovewrite = 'different'):
    '''
    Copy files and directories recursively from local path to remote path.
    Set overwrite to: 
        'different' to overwrite files with a different size;
        'always' to overwrite all files
        'never' to never overwrite
    '''
    # Create the destination directory
    if not os.path.exists(local_path):
        os.mkdir(local_path)
        #print('Local directory created:', local_path)
    
    # Connect to remote:
    sftp = connect_sftp(hostname, username, password, os.path.join(local_path, 'scp.log'))

    try:
        sftp.put_path(local_path, remote_path, ovewrite)            
        
    except: 
        sftp.close()
        
        info = exc_info()
        print_exception(*info)
        
    finally:
        sftp.close()
        
def connect_sftp(hostname, username, password = None, log_file = None):
    '''
    Make an SFTP connection. Set log_file to a valid filename to dump the log file of the connection.
    Returns:
        sftp = _MySFTPClient_
    '''
    # Create log file:
    if log_file:
        paramiko.util.log_to_file(log_file)

    # Open a transport:
    transport = paramiko.Transport((hostname, 22))
    
    # Auth
    print('Authorizing sftp connection...')
    if password:    
        transport.connect(username = username, password = password)
    else:
        transport.connect(username = username)
    
    print('Done!')
    
    client = _MySFTPClient_.from_transport(transport)
    
    # Go!
    return client 

def delete_local(local_path):
    '''
    Often useful to delete a directory recursively. Use with extreme care!
    '''    
    print('Deleting:', path)
    shutil.rmtree(path)

class _MySFTPClient_(paramiko.SFTPClient):
    '''
    Class needed for copying recursively through ssh (paramiko.SFTPClient only allows to copy single files).
    '''
    _total_file_count_ = 0
    _current_file_count_ = 0
    
    def sftp_walk(self, remote):
        '''
        From https://gist.github.com/johnfink8/2190472
        '''
        path=remote
        files=[]
        folders=[]
        for f in self.listdir_attr(remote):
            if stat.S_ISDIR(f.st_mode):
                folders.append(f.filename)
            else:
                files.append(f.filename)
        
        yield path,folders,files
        for folder in folders:
            new_path=os.path.join(remote,folder)
            for x in self.sftp_walk(new_path):
                yield x
    
    def _put_path_(self, local, remote, overwrite):
        '''
        Recursive function for uploading directories.
        '''
        if not self._exists_remote_(remote):
            #print('*making:', remote)
            self.mkdir(remote, ignore_existing=True)
        
        # Loop with a progress bar:
        pbar = tqdm(total=self._total_file_count_)
        
        for item in os.listdir(local):
            if os.path.isfile(os.path.join(local, item)):
                
                # Copy the file:
                self._current_file_count_ += 1
                
                # Overwrite if need to:
                if self._overwrite_(os.path.join(local, item), os.path.join(remote, item), overwrite):
                    self.put(os.path.join(local, item), os.path.join(remote, item))
                
                pbar.update(1)
                
            else:
                
                #print('making:', os.path.join(remote, item))
                #self.mkdir(os.path.join(remote, item), ignore_existing=True)
                self._put_path_(os.path.join(local, item), os.path.join(remote, item))
    
        pbar.close()
        
    def put_path(self, local, remote, overwrite = 'different'):
        ''' Uploads the contents of the local directory to the remote path. The
            target directory needs to exists. All subdirectories in local are 
            created under remote.
        '''
        # Count all files:
        print('Counting files...')
        self._total_file_count_ = 0
        for root, subdirs, files in os.walk(local): self._total_file_count_ += len(files)
        
        print('Uploading %u files' % self._total_file_count_)
        
        # Upload files recursively:
        self._current_file_count_= 0
        self._put_path_(local, remote)

    def _get_path_(self, local, remote, overwrite):
        """
        Recursive get method.
        """      
        
        # Create new dirs:
        if not os.path.exists(local):
            os.mkdir(local)
            #print('Local directory created:', local)
        
        # Progress bar:
        pbar = tqdm(total=self._total_file_count_)
        
        # Copy files:
        for filename in self.listdir(remote):

            if stat.S_ISDIR(self.stat(os.path.join(remote, filename)).st_mode):
                
                # uses '/' path delimiter for remote server
                self._get_path_(os.path.join(local, filename), os.path.join(remote, filename))
                
            else:
                self._current_file_count_ += 1
                
                # Overwrite if need to:
                if self._overwrite_(os.path.join(local, filename), os.path.join(remote, filename), overwrite):
                    
                    # Actual get has remote first:
                    self.get(os.path.join(remote, filename), os.path.join(local, filename))
                    
                    pbar.update()
                    
        pbar.close()
                
    def get_path(self, local, remote = 'different'):
        '''
        Download the content of the remote to the local path.
        '''
        if not self._exists_remote_(remote):
            print('Remote path doesnt exist :(((')
            return
        
        # Count all files:
        print('Counting files...')
        self._total_file_count_ = 0
        for root, subdirs, files in self.sftp_walk(remote): self._total_file_count_ += len(files)
        
        print('Downloading %u files...' % self._total_file_count_)
        
        # Upload files recursively:
        self._current_file_count_= 0
        self._get_path_(local, remote)   

    def mkdir(self, path, mode=511, ignore_existing=False):
        ''' Augments mkdir by adding an option to not fail if the folder exists  '''
        try:
            super(_MySFTPClient_, self).mkdir(path, mode)
            
        except IOError:
            if ignore_existing:
                pass
            else:
                raise
                  
    def _overwrite_(self, from_path, to_path, overwrite):
        '''
        Check if file should be overwritten.
        '''                
        if overwrite == 'never':
            return False
        elif overwrite == 'always':
            return True
        elif overwrite == 'different':
            return (self._size_local_(from_path) != self._size_remote_(to_path))
        else:
            raise Exception('Unknown overwrite mode:' + overwrite)
            
    def _size_local_(self, path):
        try:
            sta = os.stat(path)
            
            return sta.st_size
        
        except IOError as e:
            if e.errno == errno.ENOENT:
                return 0
            raise
        
    def _size_remote_(self, path):
        try:
            sta = self.stat(path)
            
            return sta.st_size
        
        except IOError as e:
            if e.errno == errno.ENOENT:
                return 0
            raise
            
    def _exists_remote_(self, path):
        try:
            self.stat(path)
        except IOError as e:
            if e.errno == errno.ENOENT:
                return False
            raise
        else:
            return True