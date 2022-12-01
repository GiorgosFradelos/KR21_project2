import copy
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

    def marginalization(self):
        return True

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








