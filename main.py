import BNReasoner as reas

import pandas as pd



''' Group 5 - Start '''
#filename = 'testing/dog_problem.BIFXML'
#filename = 'testing/alarm.BIFXML'
filename = 'testing/alarm_2.BIFXML'
#filename = 'testing/lecture_example.BIFXML'
#filename = 'testing/lecture_example2.BIFXML'


''' Create bayesian network object '''
new_BN = reas.BNReasoner(filename)
#new_BN.bn.draw_structure()

#edges = new_BN.bn.edges


''' Use Bayesian Network class functions to test their functionality '''
''' NOT NEEDED ANYMORE '''


cpts = new_BN.bn.get_all_cpts()


for var, cpt in cpts.items():
    print(cpt)

print('\n\n\n')


#new_BN.prune(query_vars, ev)
#new_BN.bn.draw_structure()

query_vars = ['B']
evidence= {'J': True, 'M': True}

index = list(evidence.keys())
ev = pd.Series(data=evidence, index=index)









#print(f'\n{x_y[0]} _|_ {x_y[1]}\t|\t{e} : {new_BN.d_sep(x_y[0], x_y[1], e)}')

#new_BN.prune(query_vars, ev)
#new_BN.bn.draw_structure()

cpts = new_BN.bn.get_all_cpts()
variables = new_BN.bn.get_all_variables()
#print(f'{variables}')

order = 3
final_cpt = new_BN.marginal_distribution(query_vars, ev, order)
print(f'\nFINAL CPT!!!:\n{final_cpt}')


