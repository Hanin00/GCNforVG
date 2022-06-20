import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import requests
import json
import sys
import time
from collections import Counter
import networkx as nx
import pickle




G1 = nx.cycle_graph(4)
G2 = nx.cycle_graph(6)

# G1.add_nodes_from([(2, {"color": "white"}), (3, {"color": "green"})])
# G2.add_nodes_from([(0, {"color": "blue"}), (3, {"color": "green"})])

#G2.remove_edge(0,1)

#mapping = {0: "a", 1: "b", 2: "c"}
mapping = {0 : 7, 1:8, 2: 9, 3:10, 4:11, 5:12}
G3 = nx.relabel_nodes(G2, mapping)
# nx.draw(G3,with_labels=True)
# plt.savefig("./data/G3.png")
G2.add_edges_from([(0, 3, {"color": "white"}), (1, 3, {"color": "green"})])
G3.add_edges_from([(7, 8, {"color": "white"}), (8, 9, {"color": "green"})])
print(nx.graph_edit_distance(G1, G2))
print(nx.graph_edit_distance(G2, G3))

#print(nx.nodes(G2))
#print(nx.nodes(G3))
print(nx.edges(G2))
#print(nx.edges(G3))
print(G3.edges())
print(type(G3.edges()))
































def ged(g1, g2, algo, debug=False, timeit=False):
    # https://github.com/dan-zam/graph-matching-toolkit
    gp = get_gmt_path()
    append_str = get_append_str(g1, g2)
    src, t_datapath = setup_temp_data_folder(gp, append_str)
    meta1 = write_to_temp(g1, t_datapath, algo, 'g1')
    meta2 = write_to_temp(g2, t_datapath, algo, 'g2')
    if meta1 != meta2:
        raise RuntimeError(
            'Different meta data {} vs {}'.format(meta1, meta2))
    prop_file = setup_property_file(src, gp, meta1, append_str)
    rtn = []
    # print(gp)
    # print(get_root_path())
    # print(append_str)
    if not exec(
            'cd {} && java {}'
            ' -classpath {}/src/graph-matching-toolkit/graph-matching-toolkit.jar algorithms.GraphMatching '
            './properties/properties_temp_{}.prop'.format(
                gp, '-XX:-UseGCOverheadLimit -XX:+UseConcMarkSweepGC -Xmx100g'
                if algo == 'astar' else '', get_root_path(), append_str)):
        rtn.append(-1)
    else:
        d, result_file = get_result(gp, algo, append_str)
    '''
        rtn.append(d)
        if g1size != g1.number_of_nodes():
            print('g1size {} g1.number_of_nodes() {}'.format(g1size, g1.number_of_nodes()))
        assert (g1size == g1.number_of_nodes())
        assert (g2size == g2.number_of_nodes())
    if debug:
        rtn += [lcnt, g1, g2]
    if timeit:
        rtn.append(t)
    clean_up(t_datapath, prop_file, result_file)
    if len(rtn) == 1:
        return rtn[0]
    return tuple(rtn)
    '''
    clean_up(t_datapath, prop_file, result_file)
    return d
