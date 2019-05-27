# model-learner
This project implements the intent discovery module of BRASS MARS project.



# Install

```bash
git clone https://github.com/cmu-mars/model-learner.git
cd model-learner
make
```

# Usage

Here an example how to use the modules in the `learner` package to machine learn a dimensional model:
```python
from learner.mlearner import MLearner
from learner.model import genModelTermsfromString, Model, genModelfromCoeff

ndim = 20
budget = 1000
model_txt = """10 + 1.00 * o0 + 2.00 * o1 + 3.00 * o2 +
4.00 * o3 + 5.00 * o4 + 6.00 * o5 + 7.00 * o6 + 8.00 * o7 + 
1.00 * o8 + 2.00 * o9 + 3.00 * o10 + 4.00 * o11 + 5.00 * o12 + 
6.00 * o13 + 7.00 * o14 + 8.00 * o15 + 1.00 * o16 + 2.00 * o17 + 
3.00 * o18 + 4.00 * o19 + 1 * o0 * o1"""

power_model_terms = genModelTermsfromString(model_txt)
true_power_model = Model(power_model_terms, ndim)
learner = MLearner(budget, ndim, true_power_model)
learned_model = learner.discover()
learned_power_model_terms = genModelfromCoeff(learned_model.named_steps['linear'].coef_, ndim)
learned_power_model = Model(learned_power_model_terms, ndim)
learned_power_model.__str__()
```

# Maintainer

If you need a new feature to be added, please contact [Pooyan Jamshidi](https://pooyanjamshidi.github.io).