from learner.model import genModel, Model, Term
import os
import numpy as np

model_path = os.path.expanduser("~/cp1/models/temp")
model_name = 'model'

scale = 100.0
N = 100
ndim = 20
max_coeff = 10

opt = 20
interactions = list(range(1, N+1))

xTest = np.ones(shape=(1, ndim))

idx = 0
for interaction in interactions:
    model_terms = genModel(opt, interaction, max_coeff)
    model = Model(model_terms, ndim)
    max_val = model.evaluateModelFast(xTest)[0]

    # Normalize model
    model_terms_normalized = []
    for term in model_terms:
        new_coeff = term.coefficient/max_val*scale
        new_term = Term(new_coeff, term.options)
        model_terms_normalized.append(new_term)

    model_normalized = Model(model_terms_normalized, ndim)

    model_txt = model_normalized.__str__()
    with open(os.path.join(model_path, model_name + str(idx)), 'w') as file:
        file.write(model_txt)
    idx += 1
