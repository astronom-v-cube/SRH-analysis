import urllib.request
from analise_utils import OsOperations
from tqdm import tqdm 

date = '202405'
OsOperations.create_place(date)

for i in tqdm(range(1, 31 + 1)):
    try:
        urllib.request.urlretrieve(f"https://badary.iszf.irk.ru/corrPlots/{date[:4]}/fFluxPlot_0612_{date}{i:02d}.png", f"{date}/{i:02d}.png")
    except Exception as err:
        tqdm.write(f'{i}: {err}')
        pass
    