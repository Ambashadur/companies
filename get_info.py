import pandas as pd
import requests as rq
import bs4

data = pd.read_csv('sites.csv', index_col='Unnamed: 0')
information = pd.DataFrame(columns=['site', 'info'])

for i in data.index:
    site = data.loc[i]

    try:
        if site['ref_to_about'] == '-':
            response = rq.get(site['site'], verify=False, timeout=5)
        else:
            response = rq.get(site['ref_to_about'], verify=False, timeout=5)

        if response.status_code == 200:
            soup = bs4.BeautifulSoup(response.content, 'html.parser')
            all_text = soup.find_all(['p', 'span'], text=True)

            raw_string = ' '.join([x.text for x in all_text])
            information = information.append({
                'site': site['site'],
                'info': raw_string
            }, ignore_index=True)
    except:
        information = information.append({
            'site': site['site'],
            'info': '-'
        }, ignore_index=True)

information.to_csv('information.csv')
