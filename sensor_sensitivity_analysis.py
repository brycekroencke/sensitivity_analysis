"""
Sensitivity Analysis for determining the ranking of sensors
"""

import numpy as np
import argparse
import imutils
import pickle
import pandas
import json
import time
import cv2
import csv
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", help="path to the input csv file")
ap.add_argument("-o", "--output", help="path to store graphs")
args = vars(ap.parse_args())

df = pandas.read_csv(args["input"])
print(df.head(5))

test_set_of_sensors(**kwargs):
    
