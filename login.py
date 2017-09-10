# -*- coding: utf8 -*-
import requests
import datetime
import time
payload = {
	'ecRequestToken':'a96d4ed0-d648-4455-a9be-ddd41cf7a99e',
	'j_username':'',
	'j_password':''
}
s = requests.session()
s.post(url='https://ectuary.insnergy.com/j_spring_security_check',data=payload)
r=s.get('https://ectuary.insnergy.com/ec2401.do')

dataparam = {
	'go':'getWeatherHistoryData',
	'sensorID':'GI21IICWB-0006010101',
	'startTime':'1473523200000',
	'endTime':'1473609600000',
	'attr':'irradiance',
	'_':'1473154271820'
}

i = 0
while i < 31:
	data = s.get('https://ectuary.insnergy.com/ec2401.do', params=dataparam)
	jsondata = data.json()

	stime = time.gmtime((int(dataparam['endTime']))/1000)
	
	if stime.tm_mon < 10:
		if stime.tm_mday < 10:
			f = open('2016'+'-'+'0'+str(stime.tm_mon)+'-'+'0'+str(int(stime.tm_mday))+'.csv', 'w')
		else:
			f = open('2016'+'-'+'0'+str(stime.tm_mon)+'-'+str(int(stime.tm_mday))+'.csv', 'w')
	else:
		if stime.tm_mday < 10:
			f = open('2016'+'-'+str(stime.tm_mon)+'-'+'0'+str(int(stime.tm_mday))+'.csv', 'w')
		else:
			f = open('2016'+'-'+'0'+str(stime.tm_mon)+'-'+str(int(stime.tm_mday))+'.csv', 'w')
	
	for set in jsondata['irradiance']:
		setdate = time.strftime("%H", time.localtime(int(set['reportTime'])/1000))
		f.write('"{temp:.2f}", "{unit}", "{date}",\n'.format(temp = float(set['value']), unit = u'W/m2', date = setdate))
	
	dataparam['startTime'] = dataparam['endTime']
	dataparam['endTime'] = str(int(dataparam['endTime']) + 86400000)
	
	f.close()
	i += 1
