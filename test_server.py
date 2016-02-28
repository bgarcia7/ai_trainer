import urllib, urllib2
import os
import json

url = 'http://54.187.178.85:5000/analyze_raw'

with open("data/raw_data/squat_pushupData_10to20/squatData13.txt") as data_file:
	values = data_file.read()



data = json.dumps({'data': values})

req = urllib2.Request(url, data, {'Content-Type': 'application/json'})

response = urllib2.urlopen(req)
the_page = response.read()
print the_page
