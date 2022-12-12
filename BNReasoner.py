import copy
import random
from typing import Union
import pandas as pd
from BayesNet import BayesNet
from typing import Union, List, Tuple, Dict, Set
import networkx as nx
import itertools
from itertools import combinations

import parameters





def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))

class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
            self.edges = self.bn.edges


        else:
            self.bn = net




    # TODO: This is where your methods should go

    def prune(self, Q: Set[str], ev: pd.Series):

        print(f'\nQuery:\n {Q}')
        print(f'\nEvidence:\n {ev}')

        if Q != {}:
            # node prune
            ev_nodes = set(ev.index)
            not_elim = ev_nodes.union(Q)
            # not_elim = Q.union(ev_nodes)
            elim = set(self.bn.get_all_variables()).difference(not_elim)
            for node in elim:
                if len(self.bn.get_children(node)) == 0:
                    self.bn.del_var(node)

        # edge-prune
        for node in list(ev.index):
            # self.bn.update_cpt(node, self.bn.get_compatible_instantiations_table(ev, self.bn.get_cpt(node)))
            for child in self.bn.get_children(node):
                edge = (node, child)
                self.bn.del_edge(edge)
                cr_d = {node: ev[node]}
                cr_ev = pd.Series(data=cr_d, index=[node])
                self.bn.update_cpt(child, self.bn.get_compatible_instantiations_table(cr_ev, self.bn.get_cpt(child)))
                self.bn.update_cpt(child, self.bn.get_cpt(child).drop(columns=[node]))

    def orderWidth(self, ordering: list) -> int:
        interaction_graph = self.bn.get_interaction_graph()
        w = 0

        for i in range(1, len(ordering)):
            d = len([k for k in interaction_graph.neighbors(ordering[i])])
            w = max(w, d)

            neighbors = interaction_graph.neighbors(ordering[i])
            interaction_graph.remove_node(ordering[i])

            for j in powerset(neighbors):
                if len(j) >= 2 and j[-1] is not None:
                    # print(j)  # for debugging
                    # print('created edges between', j, f'after removing {ordering[i]}')  # just for debugging.
                    interaction_graph.add_edge(j[0], j[1])

        return w


    def d_sep(self, X: List[str], Y: List[str], Z: List[str]):

        bn_copy = copy.deepcopy(self.bn)
        still_pruning = True
        while still_pruning:
            still_pruning = False
            # delete leaf nodes
            for node in bn_copy.get_all_variables():
                if node not in X and node not in Y and node not in Z and len(bn_copy.get_children(node)) == 0:
                    bn_copy.del_var(node)
                    still_pruning = True

            # del outgoing edges
            for ev in Z:
                children = bn_copy.get_children(ev)
                for child in children:
                    edge = (ev, child)
                    bn_copy.del_edge(edge)
                    still_pruning = True

        # list of sets with connected parts of graph
        connections = list(nx.connected_components(nx.Graph(bn_copy.structure)))
        xy = set(X).union(set(Y))
        for con in connections:
            if con.issuperset(xy):
                print("not d-separated")
                return False
        print("d-separated")
        return True






    def randomOrder(self, X: list) -> list:
        random.shuffle(X)
        return X

    def minDegreeOrder(self, X: list) -> list:
        interaction_graph = self.bn.get_interaction_graph()
        pi = list()
        nodes = dict()


        # counts all the edges to see which node has the least
        for i in X:
            for j in interaction_graph.neighbors(i):
                if j in nodes:
                    nodes[j] += 1
                else:
                    nodes[j] = 1

        print(f'\nX: {X}\n')
        print(f'Nodes: {nodes}\n')
        print(f'\n')


        for k in range(len(X)):
            var_least_edges = min(nodes, key=nodes.get)  # gets the var with the least edges
            pi.append(var_least_edges)  # puts it in the output list; lists are ordered so they're good for this
            nodes.pop(var_least_edges)  # removes this var from the counting dict

            # if there is only 1 neighbour, then just remove the node without creating any more edges
            if len(list(interaction_graph.neighbors(var_least_edges))) < 2:
                interaction_graph.remove_node(var_least_edges)
                continue

            # makes a list of neighbours
            neighbors = list(interaction_graph.neighbors(var_least_edges))
            # removes the node from the interaction graph
            interaction_graph.remove_node(var_least_edges)
            # iterates over the neighbours to create pairs and then create edges between the nodes
            for i in powerset(neighbors):
                if len(i) == 2 and not interaction_graph.has_edge(i[0], i[1]):
                    interaction_graph.add_edge(i[0], i[1])

        return pi

    def minFillOrder(self, X: list) -> list:
        interaction_graph = self.bn.get_interaction_graph()
        pi = list()
        #nodes = dict()

        X_copy = X.copy()



        for m in range(len(X)):
            # calculate the score for every variable
            nodes = dict()
            for i in X_copy:
                # print(i, len(list(interaction_graph.neighbors(i))))
                degree = len(list(interaction_graph.neighbors(i)))
                y = 0
                for j in [j for j in combinations(list(interaction_graph.neighbors(i)), 2)]:
                    if not interaction_graph.has_edge(j[0], j[1]):
                        y += 1

                nodes[i] = (((degree - 1) * degree) / 2) - y
                # print(nodes)
                # print()

            min_edges_created = min(nodes, key=nodes.get)
            pi.append(min_edges_created)

            # removes the variable from the list and the dict
            nodes.pop(min_edges_created)
            X_copy.remove(min_edges_created)

            # makes a list of neighbours
            neighbors = list(interaction_graph.neighbors(min_edges_created))
            # removes the node from the interaction graph
            interaction_graph.remove_node(min_edges_created)
            # iterates over the neighbours to create pairs and then create edges between the nodes
            for k in powerset(neighbors):
                if len(k) == 2 and not interaction_graph.has_edge(k[0], k[1]):
                    # print(k)
                    # print('created edges between', k, f'after removing {min_edges_created}')
                    interaction_graph.add_edge(k[0], k[1])

        return pi



    def summing_out(self, CPT, index_same, list):
        print(f'\n\n\n\n\n//_--------------------_//')
        print(f'list summing_out gets as Input:\n{list}')
        print(f'\nCPT summing_out gets as Input:\n{CPT}')
        # create the final new CPT without the variables that should be summed out
        new_CPT = CPT.copy()
        for variable in list:
            new_CPT = new_CPT.drop(columns=[variable])

        # loop through the keys from the dictionary
        for key in index_same:

            # pick the p_value from that key
            p_value_sum = CPT.iloc[key]['p']

            # pick the values from the key, so the equal indexes
            equal_indexes = [index_same.get(key)]

            # loop through the equal indexes
            for i in equal_indexes:
                # add this p-value from this row to the total p-val
                p = CPT.iloc[i]['p'].values
                p_value_sum += p[0]

                # set this value to zero to be able to delete it later
                new_CPT.at[i[0], 'p'] = 0


            # set the new_CPT key value to the new p value
            new_CPT.at[key, 'p'] = p_value_sum

        print(f'\nCPT after zeroing the not needed elements\n{new_CPT}')

        for index_CPT in range(len(new_CPT)):
            if new_CPT.iloc[index_CPT]['p'] == 0:
                new_CPT.drop([index_CPT])

        new_CPT = new_CPT[new_CPT.p != 0]
        print(f'\nCPT after reducing rows with p=0\n{new_CPT}')
        print(f'//-____________________-//\n\n\n\n\n')

        return new_CPT

    def maxing_out(self, CPT, index_same):
        """
        # function to sum all the values from a list out
        """

        new_CPT = CPT.copy()

        # for variable in list:
        #     new_CPT = new_CPT.drop(columns=[variable])

        # loop through the keys from the dictionary
        for key in index_same:

            # pick the p_value from that key
            p_value_max = CPT.iloc[key]['p']

            # pick the values from the key, so the equal indexes
            equal_indexes = index_same.get(key)

            # loop through the equal indexes
            for i in equal_indexes:

                # add the max p-value from this row to the total p-val
                if CPT.iloc[i]['p'] > p_value_max:
                    p_value_max = CPT.iloc[i]['p']

                # set this value to zero to be able to delete it later
                new_CPT.at[i, 'p'] = 0

            # set the new_CPT key value to the new p value
            new_CPT.at[key, 'p'] = p_value_max

        new_CPT = new_CPT[new_CPT.p != 0]
        print(f'\nCPT after reducing rows with p=0\n{new_CPT}')
        print(f'//-____________________-//\n\n\n\n\n')

        return new_CPT



    def check_double(self, CPT, list, type):
        """
        function to sum all the values from a list out
        CPT: pandas dataframe
        list: list
        returns: pandas dataframe
        """

        # create a dictionary for the equal rows with same values
        index_same = {}

        # create a CPT without p-values and without all the variables that should be summed out
        clean_CPT = CPT.copy()
        clean_CPT = clean_CPT.drop(columns=["p"])

        print(f'\n\n\n//-----------------------//')
        print(f'List: {list}')
        print(f'CPT tables that check_double gets as input:\n{clean_CPT}')


        for variable in list:
            clean_CPT = clean_CPT.drop(columns=[variable])

        #print(f'\nCPT tables after dropping variable[ {variable[0]} ] collumn:\n{clean_CPT}')


        # loop trough the length of rows of the clean CPT
        for row_1 in clean_CPT.iloc:
            for i in range(row_1.name + 1, len(clean_CPT)):
                row_2 = clean_CPT.iloc[i]

                # compare the different rows
                if row_1.equals(row_2):

                    # if it is still empty just add the new key with index number and index of equal row as value
                    if row_1.name in index_same:
                        index_same[row_1.name].append(i)
                    else:
                        index_same[row_1.name] = [i]

        if type == "sum":

            #print(f'\nIndex same:\n{index_same}')

            new_CPT = self.summing_out(CPT, index_same, list)

            print(f'\nCPT tables after sum out:\n{new_CPT}')

        elif type == "max":


            new_CPT = self.maxing_out(CPT, index_same)

            print(f'\nCPT tables after max out:\n{new_CPT}')

        print(f'//______________________//\n\n')
        return new_CPT












    def multiplying_factors(self, CPT_1, CPT_2):
        CPT_1 = CPT_1.reset_index(drop=True)
        CPT_2 = CPT_2.reset_index(drop=True)

        #print(f'\nCPT1: \n{CPT_1}')
        #print(f'\nCPT2: \n{CPT_2}')


        # matching columns
        columns_1 = list(CPT_1)
        columns_2 = list(CPT_2)


        #print(f'\ncolumns_1:\n{columns_1}')
        #print(f'\ncolumns_2:\n{columns_2}')


        columns = [x for x in columns_1 if x in columns_2]
        #print(f'\ncolumnss:\n{columns}')

        # create a dictionary for the equal rows with same values
        index_same = {}

        # create a CPT without p-values and without all the variables that should be summed out
        clean_CPT_1 = CPT_1.copy()
        clean_CPT_2 = CPT_2.copy()

        clean_CPT_1 = clean_CPT_1[columns]
        clean_CPT_1 = clean_CPT_1.drop(columns="p")

        clean_CPT_2 = clean_CPT_2[columns]
        clean_CPT_2 = clean_CPT_2.drop(columns="p")


        #print(f'\nclean_CPT_1:\n{clean_CPT_1}')
        #print(f'\nclean_CPT_2:\n{clean_CPT_2}')


        # loop trough the length of rows of the clean CPT
        for row_1 in clean_CPT_1.iloc:

            row_1 = row_1.replace([True], 1.0)
            row_1 = row_1.replace([False], 0.0)

            for row_2 in clean_CPT_2.iloc:
                row_2 = row_2.replace([True], 1.0)
                row_2 = row_2.replace([False], 0.0)


                # compare the different rows
                if row_1.equals(row_2):

                    # if it is still empty just add the new key with index number and index of equal row as value
                    if row_1.name in index_same:
                        index_same[row_1.name].append(row_2.name)
                    else:
                        index_same[row_1.name] = [row_2.name]
        #print(f'\nindex_saaame:-: \n{index_same}')

        new_columns = columns_1.copy()
        new_columns.remove('p')
        new_columns.extend(x for x in columns_2 if x not in new_columns)

        new_CPT = pd.DataFrame()
        merge_CPT_1 = CPT_1.copy()

        merge_CPT_2 = CPT_2.copy()


        for key, values in index_same.items():

            # merge rows
            row_1 = merge_CPT_1.iloc[key].drop("p")
            for value in values:
                # merge_CPT_2.reset_index()
                row_2 = merge_CPT_2.iloc[value].drop("p")

                difference = [name for name in new_columns if name not in columns_1]
                #print(f'Difference:: {difference}')

                new_row = pd.merge(row_1, row_2[difference], left_index=True, right_index=True, how='outer')
                #print(f'\nrow_1_mergee__: \n{new_row}')
                new_row = new_row.iloc[:, 0].fillna(new_row.iloc[:, 1])
                #print(f'\nrow_1_mergee: \n{new_row}')

                p_1_f = CPT_1.iloc[key]["p"]
                #print(f'\nP_1::: \n {p_1_f}')

                p_2_f = CPT_2.iloc[value]["p"]
                #print(f'\nP_2::: \n {p_2_f}')



                new_p = p_1_f * p_2_f

                new_row["p"] = round(new_p, 8)
                #print(f'\nnew_row: \n {new_row}')

                new_CPT = new_CPT.append(new_row, ignore_index=True)




        return new_CPT

    def sum_variable_in_factor(self, factor, variable):
        cpt = factor.copy()
        vars = (list(cpt))

        rest_of_vars = []
        for var in vars:
            if var != variable:
                rest_of_vars.append(var)

        same_index = {}
        cpt = cpt.drop(['p'], axis=1)
        cpt = cpt.drop([variable], axis=1)
        #print(f'\nCPTTTT FACTOR: \n{cpt}')
        factor_cp = cpt.copy()

        for i, row_i in cpt.iterrows():
            i_vals = row_i.values.tolist()

            for j, row_j in factor_cp.iterrows():



                j_vals = row_j.values.tolist()
                flag = False

                #print(f'\nvar: {var}')
                #print(f'Row I: \n{(row_i.values.tolist())}')
                #print(f'Row J: \n{(row_j.values.tolist())}')
                j_vals = row_j.values.tolist()

                flag = True
                for counter in range(len(i_vals)):
                    if i_vals[counter] == j_vals[counter]:
                        continue
                    else:
                        flag = False

                #print(f'J value:::: {j}')
                if flag == True:
                    if i in same_index:
                        same_index[i].append(j)
                    else:
                        same_index[i] = [j]


                '''if row_j.values.tolist() == row_i.values.tolist():
                    if flag == True:
                        
                    continue
                else:
                    flag = False
                    break'''

                #flag = True



        temp = []
        res = {}
        for key, val in same_index.items():
            if val not in temp:
                temp.append(val)
                res[key] = val

        cpt_sum = pd.DataFrame()
        sum={}
        for key, val in same_index.items():
            summed_pr = 0
            for i in val:
                #print(f'\n summed pr:: {summed_pr}')
                summed_pr += factor.iloc[i]['p']


            new_p = round(summed_pr, 8)

            clean = factor.iloc[i].drop(['p'])

            #new_row = clean.append(factor.iloc[j])
            new_row = clean
            new_row["p"] = new_p

            cpt_sum = cpt_sum.append(new_row, ignore_index=True)

        res = cpt_sum
        result_df = res.drop_duplicates(subset=['p'], keep='first')
        #print(f'\nHope Thats it:\n{result_df}')

        return result_df









    def multiply_cpts(self, cpt1, cpt2):
        new_CPT = pd.DataFrame()
        columns2 = list(cpt2)
        columns1 = list(cpt1)
        # diff = [x for x in columns1 if x in columns2]
        # print(f'{diff} DIFFF')
        for i in range(len(cpt1)):
            for j in range(len(cpt2)):
                p_1 = cpt1.iloc[i]["p"]
                p_2 = cpt2.iloc[j]["p"]

                new_p = round(p_1 * p_2, 8)

                clean = cpt1.iloc[i].drop(['p'])

                new_row = clean.append(cpt2.iloc[j])

                new_row["p"] = new_p

                new_CPT = new_CPT.append(new_row, ignore_index=True)
        final_cpt = new_CPT[new_CPT['p'] != 0]

        return final_cpt



    def marginal_distribution(self, Q, evidence, order):
        '''
        Q = variables in the network BN
        evidence = instantiation of some variables in the BN
        output = posterior marginal Pr(Q|e)
        '''

        ''' first prune the BN with respect to the evidence and the query in question '''
        self.prune(Q, evidence)


        type = parameters.type

        not_Q = [x for x in self.bn.get_all_variables() if x not in Q]
        if order == 1:
            order = set(self.minDegreeOrder(not_Q))
        elif order == 2:
            order = set(self.minFillOrder(not_Q))
        elif order == 3:
            order = set(self.randomOrder(not_Q))


        ''' get all cpts eliminating rows incompatible with evidence '''
        for ev in evidence.keys():

            cpts = self.bn.get_cpt(ev)
            cpts = self.bn.reduce_factor(evidence, cpts)

            # remove rows for evidence = false
            cpts = cpts[cpts.p != 0]
            print(f'\ncptssss: \n{cpts}')

            self.bn.update_cpt(ev, cpts)

            for child in self.bn.get_children(ev):
                cpts = self.bn.get_cpt(child)
                cpts = self.bn.reduce_factor(evidence, cpts)

                for row in range(len(cpts)):
                    if cpts.iloc[row]['p'] == 0:
                        cpts.drop([row])

                self.bn.update_cpt(child, cpts)




        ''' make CPTs of all variables in Pi not in Q '''
        S = list(self.bn.get_all_cpts().values())

        print(f'\norder:\n{order}')
        for variable in order:

            list_cpts = []
            list_goed = []

            print(f'\n\n\n---------__________----------')
            print(f'---------__________----------')


            print(f'\nS: {len(S)}')

            for cpt in S:

                print(f'\n{cpt}')

                columns = list(cpt)
                if variable in columns:
                    list_cpts.append(cpt)
                else:
                    list_goed.append(cpt)



            print(f'\n\n\n\t....')
            print(f'Variable: {variable}')


            print(f'\nlist_cpts:')
            for i in list_cpts:
                print(f'\n{i}')


            print(f'\n\nlist_goed:')
            for i in list_goed:
                print(f'\n{i}')




            if len(list_cpts) > 0:
                cpt1 = list_cpts[0]

            if len(list_cpts) == 1:
                list_goed.append(list_cpts[0])

            if len(list_cpts) > 1:
                print(f'\ncpt1:\n{cpt1}')


                for cpt2 in list_cpts[1:]:
                    print(f'\ncpt2:\n{cpt2}')

                    cpt1 = self.multiplying_factors(cpt1, cpt2)

                    cpt1 = self.sum_variable_in_factor(cpt1, variable)

                    print(f'\n\ncpt1*cpt2:\n{cpt1}')



                final_cpt = cpt1

                factor = self.check_double(final_cpt, [variable], type)
                print(f'\nfactor: {factor}')

                list_goed.append(factor)


            S = list_goed
            print(f'\nS After factor reduction:')
            for i in S:
                print(f'\n{i}')




        for i in range(0, len(S) - 1):


            if len(set(list(S[i])).intersection(set(list(S[i])))) > 1:
                cpt_new = self.multiplying_factors(S[i], S[i + 1])


            else:
                cpt_new = self.multiply_cpts(S[i], S[i + 1])
                S[i + 1] = cpt_new

            print(f'\nCPT New:\n{cpt_new}')


        final_cpt = cpt_new
        final_cpt = final_cpt[final_cpt['p'] != 0]

        for var in list(cpt_new):
            if var != "p":
                if var not in Q:
                    final_cpt = self.check_double(final_cpt, [var], type)

        normalize_factor = final_cpt['p'].sum()
        final_cpt['p'] = final_cpt['p'] / normalize_factor

        # zero
        final_cpt = final_cpt[final_cpt['p'] != 0]

        return final_cpt




    def MAP_MPE(self, Q, evidence, order):

        # if it is a MPE than map_var is the total BN with all the possible values in it
        self.prune(Q, evidence)

        variables = self.bn.get_all_variables()

        if list(set(variables) - set(Q)) == []:
            not_Q = variables
        else:
            not_Q = list(set(variables) - set(Q))

        if order == 1:
            order = set(self.minDegreeOrder(not_Q))
        elif order == 2:
            order = set(self.minFillOrder(not_Q))
        elif order == 3:
            order = set(self.randomOrder(not_Q))

        # add MAP variables last
        if len(Q) != 0:
            for i in Q:
                order.add(i)

        # get all cpts eliminating rows incompatible with evidence
        for ev in evidence.keys():

            cpts = self.bn.get_cpt(ev)
            cpts = self.bn.reduce_factor(evidence, cpts)

            for row in range(len(cpts)):
                if cpts.iloc[row]['p'] == 0:
                    cpts.drop([row])

            self.bn.update_cpt(ev, cpts)

            for child in self.bn.get_children(ev):
                cpts = self.bn.get_cpt(child)
                cpts = self.bn.reduce_factor(evidence, cpts)

                for row in range(len(cpts)):
                    if cpts.iloc[row]['p'] == 0:
                        cpts.drop([row])

                self.bn.update_cpt(child, cpts)

        # make CPTs of all variables in Pi not in Q
        S = list(self.bn.get_all_cpts().values())

        for variable in order:

            list_cpts = []
            list_goed = []

            for i in range(0, len(S)):
                columns = list(S[i])
                if variable in columns:
                    list_cpts.append(S[i])
                else:
                    list_goed.append(S[i])

            if len(list_cpts) == 1:
                list_goed.append(list_cpts[0])

            if len(list_cpts) > 0:
                cpt1 = list_cpts[0]

            if len(list_cpts) > 1:
                for cpt2 in list_cpts[1:]:
                    cpt1 = self.multiplying_factors(cpt1, cpt2)

                final_cpt = cpt1
                final_cpt = final_cpt[final_cpt['p'] != 0]

                if Q == {} or variable in Q:

                    cpt = self.check_double(final_cpt, [variable], 'max')

                else:
                    cpt = self.check_double(final_cpt, [variable], 'sum')

                cpt = cpt[cpt['p'] != 0]

                list_goed.append(cpt)

            S = list_goed


        if len(S) == 1:
            cpt_new = S[0]
        else:
            for i in range(0, len(S) - 1):
                cpt_new = self.multiply_cpts(S[i], S[i + 1])



                S[i + 1] = cpt_new
        final_cpt = cpt_new

        highest_p = 0
        final_cpt = final_cpt[final_cpt['p'] != 0]

        length_final_cpt = len(final_cpt)
        for i in range(length_final_cpt):

            if final_cpt.iloc[i]["p"] > highest_p:
                highest_p = final_cpt.iloc[i]["p"]
                final_row = final_cpt.iloc[i]

            # else:
            #     final_cpt.iloc[i]["p"] = 0
            # final_cpt = final_cpt.drop(index=[i])
        newest_cpt = pd.DataFrame()
        newest_cpt = newest_cpt.append(final_row, ignore_index=True)



        collumns2drop = []

        for column in newest_cpt:
            if column != "p":
                if column not in Q and Q != {}:
                    collumns2drop.append(column)
                    #print(f'\nColumn: \n{column}')
                    #newest_cpt = newest_cpt.drop(columns=[column])
                    #print(f'\nNewwest CPT:\n{newest_cpt}')


        collumns2drop = set(collumns2drop)
        collumns2drop = list(collumns2drop)
        for col in collumns2drop:
            newest_cpt = newest_cpt.drop([col], axis=1)


        return newest_cpt








