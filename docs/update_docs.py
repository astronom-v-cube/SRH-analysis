import os
from bs4 import BeautifulSoup

os.system('cd ..')
os.remove('docs/index.html')

os.system('python -m pdoc --html --force --output-dir docs analise_utils.py')

for root, dirs, files in os.walk('docs'):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f, "html.parser")

                # Проверка на наличие тега <link> для фавикона
                existing_favicon = soup.find("link", rel="icon")
                if not existing_favicon:
                    link_tag = soup.new_tag("link", rel="icon", href='favicon.ico')
                    if soup.head:
                        soup.head.append(link_tag)
                    else:
                        soup.append(soup.new_tag("head"))
                        soup.head.append(link_tag)

                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(str(soup))

os.rename('docs/analise_utils.html', 'docs/index.html')
