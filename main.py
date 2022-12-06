import BNReasoner as reas

import pandas as pd



''' Group 5 - Start '''
#filename = 'testing/dog_problem.BIFXML'
filename = 'testing/lecture_example.BIFXML'
#filename = 'testing/lecture_example2.BIFXML'


''' Create bayesian network object '''
new_BN = reas.BNReasoner(filename)
#new_BN.bn.draw_structure()

edges = new_BN.bn.edges


''' Use Bayesian Network class functions to test their functionality '''
''' NOT NEEDED ANYMORE '''
#x_y = [['Winter?'], ['Rain?']]
#x_y = [['Sprinkler?'], ['Wet Grass?']]
x_y = [['Winter?'], ['Slippery Road?']]


d = {'Winter?': True, 'Slippery Road?': True}
#d = {'Sprinkler?': True, 'Wet Grass?': True}

#ev = pd.Series(data=d, index=['Winter?', 'Rain?'])
ev = pd.Series(data=d, index=['Rain?'])

#query_vars = {"Wet Grass?"}
#query_vars = {"Sprinkler?", "Wet Grass?"}
query_vars = {"Winter?", "Slippery Road?"}


#new_BN.prune(query_vars, ev)
#new_BN.bn.draw_structure()

print(new_BN.bn.get_cpt("Sprinkler?"))
print("------------")
print(new_BN.bn.get_cpt("Rain?"))
print("------------")
print(new_BN.bn.get_cpt("Wet Grass?"))





#e=['Slippery Road?']
e=['Rain?']

print(f'\n{x_y[0]} _|_ {x_y[1]}\t|\t{e} : {new_BN.d_sep(x_y[0], x_y[1], e)}')

new_BN.prune(query_vars, ev)
new_BN.bn.draw_structure()

cpts = new_BN.bn.get_all_cpts()
variables = new_BN.bn.get_all_variables()
#print(f'{variables}')



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
