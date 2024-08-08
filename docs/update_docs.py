import os

os.system('cd ..')

# os.system('python -m pdoc --html --output-dir doc --math --search -logo https://raw.githubusercontent.com/astronom-v-cube/SRH-analysis/main/docs/logo.jpg ../analise_utils.py')

os.system('python -m pdoc --pdf --force --output-dir docs analise_utils.py')
os.system('python -m pdoc --html --force --output-dir docs analise_utils.py')