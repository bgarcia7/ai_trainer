import sys
import squat_separation as ss

sys.path.append('data')

#=====[ Import Data ]=====
import coordData3 as cd
import coordKeys as keys


class Personal_Trainer:

	def __init__(self):
		self.active = True

	def get_squats(self):

		#=====[ Get data from python file and place in DataFrame ]=====
		data = cd.data
		df = pd.DataFrame(data,columns=keys.columns)
		self.squats = ss.separate_squats(df, 'NeckY')
