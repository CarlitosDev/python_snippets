_summary_embeddings = bert_summarisation['mean_summary_embeddings']
if isinstance(_summary_embeddings, np.ndarray):
    summary_embeddings = _summary_embeddings
else:
    data = json.loads(_summary_embeddings)
    summary_embeddings = np.asarray(data) 
# normalise
summary_embeddings_norm = (summary_embeddings/np.linalg.norm(summary_embeddings)).reshape(1,-1)
summary_bert_extractive = bert_summarisation['bert_extractive']