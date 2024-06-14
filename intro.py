import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

diff=pd.read_csv('https://raw.githubusercontent.com/Varun2063/disease-diagnoser-and-drug-recommender/main/data/diffsydiw.csv')
sym=pd.read_csv('https://raw.githubusercontent.com/Varun2063/disease-diagnoser-and-drug-recommender/main/data/sym_t.csv')
dia=pd.read_csv('https://raw.githubusercontent.com/Varun2063/disease-diagnoser-and-drug-recommender/main/data/dia_t.csv')
drug=pd.read_csv("DATA/drug recommendation/drugs_for_common_treatments.csv")

def update_rx_otc(value):
    if value == 'Rx':
        return 'Prescription needed, consult a doctor'
    elif value == 'OTC':
        return 'Over the counter available'
    else:
        return value  # Keep the value if it's neither Rx nor OTC

# Update values in the rx_otc column
drug['rx_otc'] = drug['rx_otc'].apply(update_rx_otc)

drug.dropna(subset=['drug_name'], inplace=True)

sym = sym.dropna()
symptoms = sym['symptom'].tolist()

df = pd.read_csv("https://raw.githubusercontent.com/Varun2063/disease-diagnoser-and-drug-recommender/main/data/dia_t.csv")
df = df['diagnose'].str.split('\x0b', expand=True).merge(df, right_index=True, left_index=True).drop(columns=['diagnose'])
df.columns = [f'diagnose_{i+1}' for i in range(len(df.columns))]
df = df[["diagnose_6", "diagnose_1", "diagnose_2", "diagnose_3", "diagnose_4", "diagnose_5"]]
df.columns = ["did","diagnosis","cause_1", "cause_2", "cause_3", "cause_4"]
dia_t = df

dia_t.drop(index=dia_t.index[-1],axis=0,inplace=True)

sd_diff=diff.merge(sym, left_on='syd', right_on='syd')

dia_t['did'] = dia_t['did'].astype('float')

sd_diff=sd_diff.merge(dia_t[['did', 'diagnosis']], left_on='did', right_on='did')

def read_data(filename):

    data = pd.read_csv(filename)
    data=data.dropna(axis=0, how='any')
    data['syd'] = data['syd'].astype("category")
    data['did'] = data['did'].astype("category")
    matrix = coo_matrix((data['wei'].astype(float),
                        (data['did'].cat.codes.copy(),
                         data['syd'].cat.codes.copy())))
    return data, matrix,data.groupby(['did']).wei.sum(),data['syd'].cat.codes.copy()

data,matrix,diagnosis,symptoms=read_data('https://raw.githubusercontent.com/Varun2063/disease-diagnoser-and-drug-recommender/main/data/diffsydiw.csv')

def bm25_weight(data, K1=1.2, B=0.8):
    """ Weighs each row of the matrix data by BM25 weighting """
    # calculate idf per term (user)
    N = float(data.shape[0])
    idf = np.log(N / (1 + np.bincount(data.col)))
    # calculate length_norm per document (artist)
    row_sums = np.squeeze(np.asarray(data.sum(1)))
    average_length = row_sums.sum() / N
    length_norm = (1.0 - B) + B * row_sums / average_length
    # weight matrix rows by bm25
    ret = coo_matrix(data)
    ret.data = ret.data * (K1 + 1.0) / (K1 * length_norm[ret.row] + ret.data) * idf[ret.col]
    return ret

symptom_count = data.groupby('syd').size()

similarity = bm25_weight(matrix)

sym[sym['syd'].isin(list(diagnosis.index))]

Ur, Si, VTr = svds(bm25_weight(coo_matrix(matrix)), k=100)

VTr1 = pd.DataFrame(VTr)

Ur1 = pd.DataFrame(Ur)

Si1 = pd.DataFrame(Si)

Sddf=pd.DataFrame(cosine_similarity(Ur,VTr.T), columns=symptom_count.index, index=list(diagnosis.index))
Sddf.to_csv('Sddf.csv')

Sydi=pd.DataFrame(cosine_similarity(Ur,VTr.T))

def pred(symp_list):

    booknr_list = sym[sym['symptom'].isin(symp_list)]['syd'].tolist()

    # Initialize an empty dictionary to store cumulative similarity scores for diseases
    disease_scores = {}

    for booknr in booknr_list:
        # Calculate similarity scores for each symptom
        similar_diseases = Sddf[booknr].sort_values(ascending=False).head(7)

        # Accumulate similarity scores for diseases across symptoms
        for disease, score in similar_diseases.items():
            if disease in disease_scores:
                disease_scores[disease] += score
            else:
                disease_scores[disease] = score

    # Convert the dictionary to a DataFrame for easier manipulation and sorting
    combined_scores_df = pd.DataFrame(list(disease_scores.items()), columns=['Disease', 'Combined_Score'])

    # Sort the diseases based on their combined scores
    top_related_diseases = combined_scores_df.sort_values(by='Combined_Score', ascending=False).head(3)

    # Display top related diseases, their scores, and causes
    for index, row in top_related_diseases.iterrows():
        disease_id = row['Disease']
        combined_score = row['Combined_Score']

        disease_info = dia_t[dia_t['did'] == disease_id]
        disease_name = disease_info['diagnosis'].values[0]
        causes = disease_info[['cause_1', 'cause_2', 'cause_3', 'cause_4']].values.flatten()

        mask = drug.isin([disease_name]).any(axis=1)

        filtered_df = drug[mask]

        st.write(f"Disease: {disease_name}, Combined Score: {combined_score}")
        for i, cause in enumerate(causes, start=1):
            if (cause == None):
                pass
            else:
                st.write(f"\t Cause : {cause}")
        st.write()
        if not filtered_df.empty:
            sorted_drugs = filtered_df.sort_values(by='rating', ascending=False)

            top_2_drugs = sorted_drugs.head(2)
            st.write(f"Top 2 drugs for {disease_name} are:")
            for index, row in top_2_drugs.iterrows():
                st.write(f"{row['drug']} - {row['rx_otc']}")
        else:
            st.write(f"No drugs available for {disease_name}")