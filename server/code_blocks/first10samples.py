def firstTen(loaded_dataset, intermediate_df, description, method): 
    df = loaded_dataset
    samples = df.head(10)
    res = {
        'result' : samples.round(3).to_json(orient='table'),
        'output' : samples.round(3).to_json(orient='table'),
        'description' : description,
        'type' : method
    }
    intermediate_df.append(samples.round(3))
    return res

res = firstTen(self.current_df, self.intermediate_df, description, method)