#!/bin/bash

# This is a data preparation script that:
# 1) Copies the downloaded data from the ~/Downloads directory
# 2) Creates a header file from the original data which describes all of the columns
# 3) Removes the header from the data to be analyzed, leaving only the actual planet and star data

# Create the header, call it README.txt
cp /Users/glenn/Downloads/planets.csv ../data/
mv ../data/planets.csv ../data/planets_download.csv
head -383 ../data/planets_download.csv > ../data/README.txt

# Create the actual data set, call it planets.csv
sed '1,383d' < ../data/planets_download.csv > ../data/planets.csv
