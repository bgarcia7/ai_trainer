import urllib, urllib2
import os

url = 'http://54.187.178.85:5000/analyze_raw'

with open("data/raw_data/squat_pushupData_10to20/squatData13.txt") as data_file:
	values = data_file.read()



data = urllib.urlencode({'data': values})

print data
req = urllib2.Request(url, data)

response = urllib2.urlopen(req)
the_page = response.read()
