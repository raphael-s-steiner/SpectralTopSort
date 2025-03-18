import functools
import heapq
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydot
import os
import scipy
import sys

import scipy.optimize

def direction_incentive_constant():
    return 0.5

def inhomogenous_quadratic_form(x: np.ndarray, graph: nx.MultiDiGraph, vertex_list: list[None]):
    assert(x.size == len(vertex_list))
    
    outvalue = 0
    
    ind_dict = dict()
    for ind, vert in enumerate(vertex_list):
        ind_dict[vert] = ind
        
    for i in range(x.size):
        vert = vertex_list[i]
        for in_edge in graph.in_edges(vert):
            src = in_edge[0]
            if src in ind_dict.keys():
                outvalue += (x[ind_dict[src]] - x[i])**(2.0) / 2.0      # half because internal edge is double counted
            elif graph.nodes[src]["part"] < graph.nodes[vert]:
                outvalue += (1.0 - x[i])**(2.0)
            elif graph.nodes[src]["part"] > graph.nodes[vert]:
                print("Parent has larger part")
            else:
                print("Missing vertex of part in vertex list")
    
    for i in range(x.size):
        vert = vertex_list[i]
        for in_edge in graph.out_edges(vert):
            tgt = in_edge[1]
            if tgt in ind_dict.keys():
                outvalue += (x[ind_dict[tgt]] - x[i])**(2.0) / 2.0      # half because internal edge is double counted
            elif graph.nodes[tgt]["part"] > graph.nodes[vert]:
                outvalue += (-1.0 - x[i])**(2.0)
            elif graph.nodes[tgt]["part"] < graph.nodes[vert]:
                print("Child has smaller part")
            else:
                print("Missing vertex of part in vertex list")
                
    num_internal_edges = 0
    direction_incentive = 0
                
    for i in range(x.size):
        vert = vertex_list[i]
        for in_edge in graph.out_edges(vert):
            tgt = in_edge[1]
            if tgt in ind_dict.keys():
                num_internal_edges += 1
                direction_incentive += x[i]
                direction_incentive -= x[ind_dict[tgt]]
                
    if num_internal_edges > 0:
        direction_incentive = direction_incentive**(2.0)
        direction_incentive /= num_internal_edges
        direction_incentive *= direction_incentive_constant()
        outvalue -= direction_incentive
        
    return outvalue

def inhomogenous_quadratic_form_jac(x: np.ndarray, graph: nx.MultiDiGraph, vertex_list: list[None]):
    assert(x.size == len(vertex_list))
    
    outvalue = np.zeros_like(x)
    
    ind_dict = dict()
    for ind, vert in enumerate(vertex_list):
        ind_dict[vert] = ind
        
    for i in range(x.size):
        vert = vertex_list[i]
        for in_edge in graph.in_edges(vert):
            src = in_edge[0]
            if src in ind_dict.keys():
                outvalue[i] += 2.0 * (x[i] - x[ind_dict[src]])
            elif graph.nodes[src]["part"] < graph.nodes[vert]:
                outvalue[i] += 2.0 * (x[i] - 1.0)
            elif graph.nodes[src]["part"] > graph.nodes[vert]:
                print("Parent has larger part")
            else:
                print("Missing vertex of part in vertex list")
    
    for i in range(x.size):
        vert = vertex_list[i]
        for in_edge in graph.out_edges(vert):
            tgt = in_edge[1]
            if tgt in ind_dict.keys():
                outvalue[i] += 2.0 * (x[i] - x[ind_dict[tgt]])
            elif graph.nodes[tgt]["part"] > graph.nodes[vert]:
                outvalue[i] += 2.0 * (x[i] - (-1.0))
            elif graph.nodes[tgt]["part"] < graph.nodes[vert]:
                print("Child has smaller part")
            else:
                print("Missing vertex of part in vertex list")
                
    num_internal_edges = 0
    direction_incentive = 0
    sum_signed_edges = np.zeros_like(x)
                
    for i in range(x.size):
        vert = vertex_list[i]
        for in_edge in graph.out_edges(vert):
            tgt = in_edge[1]
            if tgt in ind_dict.keys():
                num_internal_edges += 1
                direction_incentive += x[i]
                sum_signed_edges[i] += 1.0
                direction_incentive -= x[ind_dict[tgt]]
                sum_signed_edges[ind_dict[tgt]] -= 1.0
                
    if num_internal_edges > 0:
        direction_incentive /= num_internal_edges
        direction_incentive *= direction_incentive_constant()
        outvalue -= (2.0 * sum_signed_edges * direction_incentive)
        
    return outvalue


def inhomogenous_quadratic_form_hess(x: np.ndarray, graph: nx.MultiDiGraph, vertex_list: list[None]):
    assert(x.size == len(vertex_list))
    
    outvalue = np.zeros((x.size, x.size))
    
    ind_dict = dict()
    for ind, vert in enumerate(vertex_list):
        ind_dict[vert] = ind
        
    for i in range(x.size):
        vert = vertex_list[i]
        for in_edge in graph.in_edges(vert):
            src = in_edge[0]
            if src in ind_dict.keys():
                outvalue[i][i] += 2.0
                outvalue[i][ind_dict[src]] -= 2.0
                outvalue[ind_dict[src]][i] -= 2.0
            elif graph.nodes[src]["part"] < graph.nodes[vert]:
                outvalue[i][i] += 2.0 * x[i]
            elif graph.nodes[src]["part"] > graph.nodes[vert]:
                print("Parent has larger part")
            else:
                print("Missing vertex of part in vertex list")
    
    for i in range(x.size):
        vert = vertex_list[i]
        for in_edge in graph.out_edges(vert):
            tgt = in_edge[1]
            if tgt in ind_dict.keys():
                outvalue[i][i] += 2.0
            elif graph.nodes[tgt]["part"] > graph.nodes[vert]:
                outvalue[i][i] += 2.0 * x[i]
            elif graph.nodes[tgt]["part"] < graph.nodes[vert]:
                print("Child has smaller part")
            else:
                print("Missing vertex of part in vertex list")
                
    num_internal_edges = 0
    sum_signed_edges = np.zeros(x.size)
                
    for i in range(x.size):
        vert = vertex_list[i]
        for in_edge in graph.out_edges(vert):
            tgt = in_edge[1]
            if tgt in ind_dict.keys():
                num_internal_edges += 1
                sum_signed_edges[i] += 1.0
                sum_signed_edges[ind_dict[tgt]] -= 1.0
                
    if num_internal_edges > 0:
        direction_incentive = 1.0 / num_internal_edges
        direction_incentive *= direction_incentive_constant()
        
        outer = np.outer(sum_signed_edges, sum_signed_edges)
        outer *= 2.0
        outer *= direction_incentive
        
        outvalue -= outer
    
    return outvalue

def lin_constraint(size: int):
    
    linear_constraint = scipy.optimize.LinearConstraint(np.ones(size), 0, 0)
    
    return linear_constraint

def nonlin_constraint():
    
    def constr_func(x: np.ndarray) -> np.ndarray:
        return np.sum(x**2)
    
    def constr_jac(x: np.ndarray) -> np.ndarray:
        return 2*x
    
    def constr_hess(x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return v[0] * 2 * np.eye(x.size)
    
    nonlinear_constraint = scipy.optimize.NonlinearConstraint(constr_func, 1, 1, jac=constr_jac, hess=constr_hess)
    
    return nonlinear_constraint



def spectral_split(graph: nx.MultiDiGraph, vertex_list: list[None]) -> list[list[None],list[None]]:
    if len(vertex_list) == 0:
        return [[],[]]
    if len(vertex_list) == 1:
        return [vertex_list, []]
    
    opt_func = functools.partial(inhomogenous_quadratic_form, graph=graph, vertex_list=vertex_list)
    opt_jac = functools.partial(inhomogenous_quadratic_form_jac, graph=graph, vertex_list=vertex_list)
    opt_hess = functools.partial(inhomogenous_quadratic_form_hess, graph=graph, vertex_list=vertex_list)
    
    x0 = np.random.normal(loc=0.0, scale=(10.0)**(-8.0), size=len(vertex_list))
    x0 = x0 / np.linalg.norm(x0)
    
    result = scipy.optimize.minimize(opt_func, x0, method='trust-constr', jac=opt_jac, hess=opt_hess, constraints=[lin_constraint(len(vertex_list)), nonlin_constraint()], options={'verbose': 1})
    
    earlier = []
    later = []
    
    for ind, val in enumerate(result.x):
        if val > 0:
            earlier.append(vertex_list[ind])
        else:
            later.append(vertex_list[ind])
    
    return [earlier, later]

def top_order_fix(graph: nx.MultiDiGraph, earlier: list[None], later: list[None]) -> list[None]:
    vertices = []
    vertices.extend(earlier)
    vertices.extend(later)
    
    ind_dict = dict()
    for ind, vert in enumerate(vertices):
        ind_dict[vert] = ind
    
    remaining_parents = [ 0 for v in vertices]
    priority = [ [0,0,v] for v in vertices ]
    
    num_e = len(earlier)
    for ind in range(num_e, len(vertices)):
        priority[ind][0] = 1
    
    induced_graph = nx.induced_subgraph(graph, vertices)
    for ind, vert in enumerate(vertices):
        remaining_parents[ind] = induced_graph.in_degree(vert)
        
        for edge in graph.out_edges(vert):
            src = edge[0] # =vert
            tgt = edge[1]
            if graph.nodes[src]['part'] == graph.nodes[tgt]["part"]:
                if (ind < num_e) and (ind_dict[tgt] >= num_e):
                    priority[ind][1] += 1
                if (ind >= num_e) and (ind_dict[tgt] < num_e):
                    priority[ind][1] -= 1
            elif graph.nodes[src]['part'] < graph.nodes[tgt]["part"]:
                priority[ind][1] += 1
            else:
                print("Topological order violated")
                
        for edge in graph.in_edges(vert):
            src = edge[0]
            tgt = edge[1] # =vert
            if graph.nodes[src]['part'] == graph.nodes[tgt]["part"]:
                if (ind < num_e) and (ind_dict[src] >= num_e):
                    priority[ind][1] += 1
                if (ind >= num_e) and (ind_dict[src] < num_e):
                    priority[ind][1] -= 1
            elif graph.nodes[src]['part'] < graph.nodes[tgt]["part"]:
                priority[ind][1] -= 1
            else:
                print("Topological order violated")
    
    top_ord = []
    queue = []
    heapq.heapify(queue)
    
    for ind, val in enumerate(remaining_parents):
        if val == 0:
            heapq.heappush(queue, priority[ind])
            
    while len(queue) != 0:
        el_prio, edge_prio, vert = heapq.heappop(queue)
        top_ord.append(vert)
        
        for edge in induced_graph.out_edges(vert):
            tgt = edge[1]
            index = ind_dict[tgt]
            remaining_parents[index] -= 1
            if remaining_parents[index] == 0:
                heapq.heappush(queue, priority[index])
    
    return top_ord

def spec_top_order(graph: nx.MultiDiGraph) -> list[str]:
    if (not nx.is_directed_acyclic_graph(graph)):
        print("Graph is not acyclic")
        return []
    
    nx.set_node_attributes(graph, 0, "part")
    nx.set_node_attributes(graph, 0, "part_new")
    
    # first iteration
    earlier, later = spectral_split(graph, list(graph.nodes))
    e_set = set(earlier)
    l_set = set(later)
    
    edge_diff = 0
    for edge in graph.edges:
        if (edge[0] in e_set) and (edge[1] in l_set):
            edge_diff += 1
        if (edge[0] in l_set) and (edge[1] in e_set):
            edge_diff -= 1
            
    if (edge_diff < 0):
        earlier, later = later, earlier
        
    top_order_fix(graph, earlier, later)
        

    
    return []

def main():
    if (len(sys.argv) < 2):
        print("Usage: " + sys.argv[0] + " <graph.dot>")
        return 1
        
    graph_file = sys.argv[1]
    graph = nx.nx_pydot.read_dot(graph_file)
    
    weak_comp = nx.weakly_connected_components(graph)
    
    top_order = []
    
    for comp in weak_comp:
        subgraph = nx.induced_subgraph(graph, comp)
        subgraph = subgraph.copy()
        top_order.extend( spec_top_order(subgraph) )
    
    if (len(top_order) != graph.number_of_nodes()):
        return 1
    
    

    return 0

if __name__ == "__main__":
    main()