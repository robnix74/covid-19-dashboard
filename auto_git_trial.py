import subprocess as cmd
import os

success_flag = 1

try:
	os.chdir('F:/Learnings/Plotly and Dash/Interactive Python Dashboards with Plotly and Dash/Scripts/app_deploy')
except Exception as e:
	print('OS Error : ', e)


try:
	cp = cmd.run("git add .", check=True, shell=True)
	#print(cp)

	message = 'test-update'

	cp = cmd.run("git commit -m {}".format(message), check=True, shell=True)
	#print(cp)

	cp = cmd.run("git push https://github.com/robnix74/covid-19-dashboard master", check=True, shell=True)
	#print(cp)

except Exception as e:
	success_flag = 0
	print('Auto Git Failed\n','Error Message : ',e)

# if success_flag == 1:

# 	try:
# 		print('Starting to commit to heroku master')

# 		cp = cmd.run("git push heroku master", check=True, shell=True)
# 		print(cp)

# 		cp = cmd.run("heroku ps:scale web=1")
# 		print(cp)

# 	except Exception as e:
# 		success_flag = 0
# 		print('Auto Git Heroku Failed\n','Error Message : ',e)