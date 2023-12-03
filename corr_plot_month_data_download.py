import urllib.request

date = '202303'

for i in range(1, 31 + 1):
    try:
        urllib.request.urlretrieve(f"https://badary.iszf.irk.ru/corrPlots/{date[:4]}/fFluxPlot{date}{i:02d}.png", f"{i:02d}.png")
        print(i)
    except:
        print('err')
        pass