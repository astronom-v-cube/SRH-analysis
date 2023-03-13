import urllib.request

date = '202302'

for i in range(1, 31 + 1):

    try:

        if i//10 == 0:
            urllib.request.urlretrieve(f"http://badary.iszf.irk.ru/corrPlots/{date[:4]}/fFluxPlot{str(date) + str(0) + str(i)}.png", f"00{i}.jpg")
            print(i)
        
        else:
            urllib.request.urlretrieve(f"http://badary.iszf.irk.ru/corrPlots/{date[:4]}/fFluxPlot{str(date) + str(i)}.png", f"0{i}.jpg")
            print(i)

    except:

        try:

            if i//10 == 0:
                urllib.request.urlretrieve(f"https://badary.iszf.irk.ru/corrPlots/{date[:4]}/fFluxPlot_0612_{str(date) + str(0) + str(i)}.png", f"00{i}.jpg")
                print(i)
        
            else:   
                urllib.request.urlretrieve(f"https://badary.iszf.irk.ru/corrPlots/{date[:4]}/fFluxPlot_0612_{str(date) + str(i)}.png", f"0{i}.jpg")
                print(i)

        except:
            print(f'Дата {i} не найдена')