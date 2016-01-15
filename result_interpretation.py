from collections import Counter
import sys
sys.path.append('feedback')
import advice_messages

def advice(exercise, results):
	messages = advice_messages.messages[exercise]
	categories = advice_messages.messages[exercise+'_categories']
	
	major_problems = []
	minor_problems = []
	strengths = []

	#=====[ Iterate through each component  ]=====
	for key in results:
		num_reps = len(results[key])
		reps = (Counter(results[key])).most_common()

		#=====[ Iterate through each rep  ]=====
		for counts in reps:
			
			severity = float(counts[1])/num_reps
			
			#=====[ Decide if label for rep indicates problem area  ]=====
			if counts[0] != 0:

				#=====[ Decide if label for rep indicates major problem area  ]=====
				if severity > 0.5:
					major_problems.append(key)

				elif severity > 0.25:
					minor_problems.append(key)

			else:
				
				#=====[ Decide if label indicates strong performance in category  ]=====
				if severity > 0.9:
					strengths.append(key)

	# print_problems(major_problems, messages, categories, 'Major')
	# print_problems(minor_problems, messages, categories, 'Minor')
	# print_strengths(strengths, messages, categories)
	advice = 'A total of ' + str(num_reps) + ' were analyzed'
	advice += get_problems(major_problems, messages, categories, 'Major') + '\n'
	advice += get_problems(minor_problems, messages, categories, 'Minor') + '\n'
	advice += get_strengths(strengths, messages, categories)
	return advice 
	
def get_problems(problems, messages, categories, problem_type):
	output = ''				
	if len(problems) > 0:
		output += '\n\n' + problem_type + ' Problems:\n\n'
		for problem in problems:
			output += categories[problem] + ":" + messages[problem] + '\n'
	return output

def get_strengths(strengths, messages, categories):
	output = ''
	if len(strengths) > 0:
		output += '\n\nStrengths:\n\n'
		for strength in strengths:
			output+= categories[strength] + '\n'
	return output

def print_problems(problems, messages, categories, problem_type):				
	if len(problems) > 0:
		print '\n\n' + problem_type + ' Problems:\n'
		for problem in problems:
			print categories[problem], ":", messages[problem]

def print_strengths(strengths, messages, categories):
	if len(strengths) > 0:
		print '\n\nStrengths:\n'
		for strength in strengths:
			print categories[strength]


