import os

'''
Created on 14 Nov. 2022

@author: brandon
'''


print("Hello There")

curr_dir = os.getcwd()

print ("current directory", curr_dir)

data_dir = curr_dir.replace("test", "data", 1)

print("data dir:", data_dir)