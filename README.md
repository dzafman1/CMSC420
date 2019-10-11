# Lodestar

Setup Directions
To set up Lodestar: 
	1. Download Lodestar from https://github.com/zhecui/incrementalAnalysis
	2. Cd into incrementalAnalysis and make sure you have NPM installed as well as Python
		a. NPM version should be 10 
		b. Python version can be 2 or 3
	3. Run > NPM install -g polymer-cli 
		a. This globally installs Polymer CLI on the machine
		b. If Python 3 is install, Polymer won't work correctly so you'll need to install bower 
		c. Npm install -g bower pretty much worked
	4. Cd to server/src and Run > Python Run.py
		a. This will give you a list of dependencies for the run.py file. Once you install all of these dependencies… you should be good to go! 
	5. To run the server, 'python run.py' in server/src
	6. To run the front end, 'polymer serve' in root 

To set up the Docker VM: 
	- You need to set the IP address of the backend to 0.0.0.0:5000 
You need to be able to set up the front end to receive 0.0.0.0:5000 

