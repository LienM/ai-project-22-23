import os
import pandas as pd


"""
Now that the predictions are finished, do some postprocessing 
(concat dataframes, add a zero before each article id, fill empty spots with random recommendations)
"""

if not os.path.isdir('../../data/lstm/output_decoded/part'):
    os.mkdir('../../data/lstm/output_decoded/part')

encoded = open('../../data/lstm/transaction_encoding.csv')

customer_ids = {}
article_ids = {}

lines = encoded.readlines()[1:]
for line in lines:
    # 0: index
    # 1: t_dat
    # 2: customer_id
    # 3: article_id
    # 4: article_id_encoded
    # 5: customer_id_encoded
    elements = line.split(',')
    customer_ids[elements[5][:-1]] = elements[2]
    article_ids[elements[4]] = elements[3]


all_customers = []
all_articles = []
for file in sorted(os.listdir('../../data/lstm/output')):

    input_file = open(f'../../data/lstm/output/{file}')

    lines = input_file.readlines()[1:]

    for line in lines:

        customer_id = line.split(',')[0]
        recommendations = [x.strip('\n') for x in line.split(',')[1:]]

        # 1: customer_id
        # 2 - ... recommendations

        decoded_customer_ids = customer_ids[customer_id]
        decoded_recommendations = [article_ids[article_id] for article_id in recommendations]

        all_customers.append(f'{decoded_customer_ids}')
        all_articles.append(f'{" ".join(decoded_recommendations)}')

    print(file)

df_all = pd.DataFrame(
    data={'customer_id': all_customers, 'prediction': all_articles}
)
df_random = pd.read_csv('../../data/lstm/random_recommendations.csv')

df_all_new = df_all.merge(df_random.drop_duplicates(), on=['customer_id'], how='left', indicator=True)
df_not_present = df_all_new.loc[df_all_new['_merge'] == 'left_only']

df_not_present['prediction'] = df_not_present['prediction_x']
df_not_present = df_not_present[df_all_new.columns]

df_all = pd.concat([df_all, df_not_present])
df_all.to_csv('../../data/lstm/submission_temp.csv', index=False)

df = pd.read_csv('../../data/lstm/submission_temp.csv')

file = open('../../data/lstm/submission_temp.csv')
out_file = open('submission.csv', 'w')
out_file.write(file.readline())
line = file.readline()
while line:
    line = line.replace(' ', ' 0')
    line = line.replace(',', ',0')
    out_file.write(line)
    line = file.readline()
out_file.close()

