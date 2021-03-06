import sys, os, time
import networkx as nx

def dominant(g):
    """
        A Faire:         
        - Ecrire une fonction qui retourne le dominant du graphe non dirigé g passé en parametre.
        - cette fonction doit retourner la liste des noeuds d'un petit dominant de g

        :param g: le graphe est donné dans le format networkx : https://networkx.github.io/documentation/stable/reference/classes/graph.html

    """
    def sample_max_neighbors_node(s):
        
        node = None
        m = 0
        for n in s:
            if len(g[n]) > m:
                node = n
                m = len(g[n])
        return node
    

    all_nodes = set(g)
    start_with = sample_max_neighbors_node(all_nodes)
    dominating_set = {start_with}
    dominated_nodes = set(g[start_with])
    remaining_nodes = all_nodes - dominated_nodes - dominating_set
    
    while remaining_nodes:
        v = sample_max_neighbors_node(remaining_nodes)
        remaining_nodes.remove(v)
        undominated_neighbors = set(g[v]) - dominating_set
        dominating_set.add(v)
        dominated_nodes |= undominated_neighbors
        remaining_nodes -= undominated_neighbors
    
    new_g = nx.Graph()
    for n in dominating_set:
        new_g.add_node(n)
    return new_g.nodes 

#########################################
#### Ne pas modifier le code suivant ####
#########################################
if __name__=="__main__":
    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    
    # un repertoire des graphes en entree doit être passé en parametre 1
    if not os.path.isdir(input_dir):
	    print(input_dir, "doesn't exist")
	    exit()

    # un repertoire pour enregistrer les dominants doit être passé en parametre 2
    if not os.path.isdir(output_dir):
	    print(input_dir, "doesn't exist")
	    exit()       
	
    # fichier des reponses depose dans le output_dir et annote par date/heure
    output_filename = 'answers_{}.txt'.format(time.strftime("%d%b%Y_%H%M%S", time.localtime()))             
    output_file = open(os.path.join(output_dir, output_filename), 'w')

    for graph_filename in sorted(os.listdir(input_dir)):
        # importer le graphe
        g = nx.read_adjlist(os.path.join(input_dir, graph_filename))
        
        # calcul du dominant
        D = sorted(dominant(g), key=lambda x: int(x))

        # ajout au rapport
        output_file.write(graph_filename)
        for node in D:
            output_file.write(' {}'.format(node))
        output_file.write('\n')
        
    output_file.close()
