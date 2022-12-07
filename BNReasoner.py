import copy
import random
from typing import Union
import pandas as pd
from BayesNet import BayesNet
from typing import Union, List, Tuple, Dict, Set
import networkx as nx





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

        # given set of variables Q and evidence, prunes e &n
        # -> returns edge, node-pruned BN


    def check_independence(self):
        return True




    def randomOrder(self, X: list) -> list:
        random.shuffle(X)
        return X

    def summing_out(self, CPT, index_same, list):
        print(f'\n\n\n\n\n//_--------------------_//')
        print(f'list summing_out gets as Input:\n{list}')
        print(f'CPT summing_out gets as Input:\n{CPT}')
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

        print(f'\nCPT tables after dropping variable[ {variable[0]} ] collumn:\n{clean_CPT}')


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

            print(f'\nIndex same:\n{index_same}')

            new_CPT = self.summing_out(CPT, index_same, list)

            print(f'\nCPT tables after sum out:\n{new_CPT}')

        elif type == "max":
            new_CPT = self.maxing_out(CPT, index_same)

        print(f'//______________________//\n\n')
        return new_CPT

    def multiplying_factors(self, CPT_1, CPT_2):
        CPT_1 = CPT_1.reset_index(drop=True)
        CPT_2 = CPT_2.reset_index(drop=True)

        # matching columns
        columns_1 = list(CPT_1)
        columns_2 = list(CPT_2)
        columns = [x for x in columns_1 if x in columns_2]

        # create a dictionary for the equal rows with same values
        index_same = {}

        # create a CPT without p-values and without all the variables that should be summed out
        clean_CPT_1 = CPT_1.copy()
        clean_CPT_2 = CPT_2.copy()

        clean_CPT_1 = clean_CPT_1[columns]
        clean_CPT_1 = clean_CPT_1.drop(columns="p")

        clean_CPT_2 = clean_CPT_2[columns]
        clean_CPT_2 = clean_CPT_2.drop(columns="p")

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

                new_row = pd.merge(row_1, row_2[difference], left_index=True, right_index=True, how='outer')
                new_row = new_row.iloc[:, 0].fillna(new_row.iloc[:, 1])
                p_1 = CPT_1.iloc[key]["p"]
                p_2 = CPT_2.iloc[value]["p"]
                new_p = p_1 * p_2

                new_row["p"] = round(new_p, 8)

                new_CPT = new_CPT.append(new_row, ignore_index=True)

        return new_CPT

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

        self.prune(Q, evidence)
        not_Q = [x for x in self.bn.get_all_variables() if x not in Q]
        if order == 1:
            order = set(self.minDegreeOrder(not_Q))
        elif order == 2:
            order = set(self.minFillOrder(not_Q))
        elif order == 3:
            order = set(self.randomOrder(not_Q))

        # order = set(self.randomOrder(self.bn.get_all_variables()))
        # order_no_Q = order.difference(Q)

        # get all cpts eliminating rows incompatible with evidence
        for ev in evidence.keys():

            cpts = self.bn.get_cpt(ev)
            cpts = self.bn.reduce_factor(evidence, cpts)

            for row in range(len(cpts)):
                if cpts.iloc[row]['p'] == 0:
                    cpts.drop([row])

            cpts = cpts[cpts.p != 0]
            print(f'\ncpts: {cpts}')

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

        print(f'\norder:\n{order}')
        for variable in order:

            list_cpts = []
            list_goed = []

            print(f'\n\n\n---------__________----------')
            print(f'---------__________----------')

            # to remove None values in list
            #S = [i for i in S if i is not None]
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

                    print(f'\n\ncpt1*cpt2:\n{cpt1}')

                #print(f'\n\nlist_cpts - take 2:')
                #for i in list_cpts:
                #    print(f'\n{i}')



                final_cpt = cpt1



                factor = self.check_double(final_cpt, [variable], 'sum')
                print(f'\nfactor: {factor}')

                list_goed.append(factor)


            S = list_goed
            print(f'\nS After factor reduction:')
            for i in S:
                print(f'\n{i}')


        print(f'\n\nJust before the chaos')


        for i in range(0, len(S) - 1):

            print(f'\nWhatever: {len(set(list(S[i])).intersection(set(list(S[i]))))}')

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
                    final_cpt = self.check_double(final_cpt, [var], 'sum')

        normalize_factor = final_cpt['p'].sum()
        final_cpt['p'] = final_cpt['p'] / normalize_factor

        # zero
        final_cpt = final_cpt[final_cpt['p'] != 0]

        return final_cpt






    def maxing_out(self):
        return True

    def factor_multiplication(self):
        return True

    def ordering(self):
        return True

    def variable_elimination(self):
        return True

    def map(self):
        return True

    def mep(self):
        return True








