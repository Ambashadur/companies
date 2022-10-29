import pandas as pd
import requests as rq
import bs4
from pandas import DataFrame

invalid_sites = ['-', 'Тестовый сайт', 'Не указано', 'на разработке', 'нет']

path: str = input('path to file: ')

data: DataFrame = pd.read_excel(path)
data = data.loc[
    (data['Сайт'] != '-') &
    (data['Сайт'] != 'Тестовый сайт') &
    (data['Сайт'] != 'Не указано') &
    (data['Сайт'] != 'на разработке') &
    (data['Сайт'] != 'нет')]

sites = DataFrame(columns=['site', 'ref_to_about'])

for index in data.index:
    print(f'Process for index {index}')

    entity = data.loc[index]

    print(f'-- {entity["Сайт"]} - global id: {entity["global_id"]} ---')

    try:
        response = rq.get(entity["Сайт"], verify=False, timeout=2)

        if response.status_code == 200:
            soup = bs4.BeautifulSoup(response.content, 'html.parser')
            refs = soup.find_all('a', href=True, text=['О компании', 'About us', 'About company'])

            if len(refs) > 0:
                ref = refs[0]['href']

                if 'http' in ref:
                    sites = sites.append({
                        'site': entity["Сайт"],
                        'ref_to_about': ref
                    }, ignore_index=True)
                elif ref[0] == '/':
                    sites = sites.append({
                        'site': entity["Сайт"],
                        'ref_to_about': entity["Сайт"] + ref
                    }, ignore_index=True)
                else:
                    sites = sites.append({
                        'site': entity["Сайт"],
                        'ref_to_about': entity["Сайт"] + '/' + ref
                    }, ignore_index=True)
            else:
                sites = sites.append({
                    'site': entity["Сайт"],
                    'ref_to_about': '-'
                }, ignore_index=True)
    except:
        print(f'Failure for site: {entity["Сайт"]}')

sites.to_csv('sites.csv')
