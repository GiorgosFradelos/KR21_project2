import BNReasoner as reas

import pandas as pd
import parameters


''' Group 5 - Start '''

''' Get flename from parameters file '''
filename = parameters.filename

''' Create bayesian network object '''
new_BN = reas.BNReasoner(filename)
#new_BN.bn.draw_structure()

#edges = new_BN.bn.edges



cpts = new_BN.bn.get_all_cpts()
for var, cpt in cpts.items():
    print(cpt)

print('\n\n\n')



''' Get query variables and evidence from parameters file '''
query_vars = parameters.query_vars
evidence= parameters.evidence

index = list(evidence.keys())
ev = pd.Series(data=evidence, index=index)





#cpts = new_BN.bn.get_all_cpts()
#variables = new_BN.bn.get_all_variables()
#print(f'{variables}')

order = parameters.order

final_cpt = new_BN.marginal_distribution(query_vars, ev, order)
print(f'\nFINAL CPT!!!:\n{final_cpt}')

mpe = new_BN.MAP_MPE(query_vars, ev, order)

print(f'Query vars: {query_vars}')
if query_vars == []:
    print(f'\nGiven \n{ev}')
    print(f'Most probable explanation:\n{mpe}')
else:
    print(f'\nGiven \n{ev}')
    print(f'\nMaximum a-posteriory instantiation:\n{mpe}')



