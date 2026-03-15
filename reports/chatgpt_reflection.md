# ChatGPT Usage Reflection

## Arguing Against ChatGPT - Case 1

ChatGPT initially tends to suggest adding more model complexity or more feature sources. For this assignment, I rejected that direction because the project constraints prioritize tabular models, interpretability, and a reliable end-to-end product over complexity. A smaller, well-evaluated system is easier to defend than a larger system with weak validation.

## Arguing Against ChatGPT - Case 2

ChatGPT often suggests generic train/test splitting. I rejected that and used a spatial hold-out strategy because neighboring cells are correlated. A random split would likely overestimate performance and make the reported results less trustworthy.
