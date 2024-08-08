import os

os.system('cd ..')
os.remove('docs/index.html')

# os.system('python -m pdoc --pdf --force --output-dir docs analise_utils.py')
os.system('python -m pdoc --html --force --output-dir docs analise_utils.py')

os.rename('docs/analise_utils.html', 'docs/index.html')
