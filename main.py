import BNReasoner as reas

import pandas as pd



''' Group 5 - Start '''
#filename = 'testing/dog_problem.BIFXML'
filename = 'testing/alarm.BIFXML'
#filename = 'testing/lecture_example.BIFXML'
#filename = 'testing/lecture_example2.BIFXML'


''' Create bayesian network object '''
new_BN = reas.BNReasoner(filename)
#new_BN.bn.draw_structure()

#edges = new_BN.bn.edges


''' Use Bayesian Network class functions to test their functionality '''
''' NOT NEEDED ANYMORE '''





#new_BN.prune(query_vars, ev)
#new_BN.bn.draw_structure()

query_vars = ['B']
evidence= {'J': True, 'M': False}

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

'''vars = []
for var in variables:
    vars.append(pd.DataFrame(cpts[var]))


winter = pd.DataFrame(cpts['Winter?'])
sprinkler = pd.DataFrame(cpts['Sprinkler?'])
rain = pd.DataFrame(cpts['Rain?'])
wet_grass = pd.DataFrame(cpts['Wet Grass?'])
slip_road = pd.DataFrame(cpts['Slippery Road?'])

print(f'\n\n{winter}')
print(f'\n\n{sprinkler}')
print(f'\n\n{rain}')
print(f'\n\n{wet_grass}')
print(f'\n\n{slip_road}')


inst = pd.Series({"Winter?": True, "Rain?": False})
new = new_BN.bn.get_compatible_instantiations_table(inst, wet_grass)'''
