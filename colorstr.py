# from ipdb import set_trace
def colstr(string, color, best):
	# set_trace()
	if color is None:
		cstring = string
	elif color == 'red':
		cstring = "\033[31m" + string  + "\033[0m"
	elif color == 'green':    
		cstring = "\033[32m" + string  + "\033[0m"

	if best: 
		cstring += " ** "
	else:
		cstring += "    "

	return cstring    