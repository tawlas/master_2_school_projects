import sys, os, time
import networkx as nx

# calcul de distance de manhattan entre 2 points, un point est un tuple (x, y)
def manhatan_distance (p_1, p_2):
    return abs(p_1[0] - p_2[0]) + abs(p_1[1] - p_2[1]) 

def offline_k_serveurs(pos_sites, k, demandes):
    """
        A Faire:         
        - Ecrire une fonction qui retourne la liste des serveurs répondant aux demandes 

        :param pos_sites: un dictionnaire Python représentant la position de chaque site où:
            -- la clé (int) est le numéro du site entre 0 et n-1 (on a n sites)
            -- la valeur est un tuple (x, y) de coordonnées cartésiennes
        
        :param k: nombre de serveur dispos au départ à la position initiale (0,0). On numérotera ainsi les serveurs de 0 à k-1
        
        :param demandes: une liste Python représentant la liste des demandes sous forme d'une liste de numéros de sites. 
                Sa taille correspond aux N demandes et ses valeurs sont des entiers entre 0 et n-1

        :return la liste des numéros de serveurs répondant aux demandes, elle aura la même taille N que la liste des demandes passée en paramètre.
                Les valeurs de cette liste sont des entiers entre 0 et k-1          
    """
    return [0 for _ in demandes] # ceci n'est pas une solution optimale

##############################################################
#### LISEZ LE MANIFEST et NE PAS MODIFIER LE CODE SUIVANT ####
##############################################################
if __name__=="__main__":
    input_dir = os.path.abspath(sys.argv[1])
    output_dir = os.path.abspath(sys.argv[2])
    
    # un repertoire des graphes en entree doit être passé en parametre 1
    if not os.path.isdir(input_dir):
	    print(input_dir, "doesn't exist")
	    exit()

    # un repertoire pour enregistrer les dominants doit être passé en parametre 2
    if not os.path.isdir(output_dir):
	    print(output_dir, "doesn't exist")
	    exit()       
	
    # fichier des reponses depose dans le output_dir et annote par date/heure
    output_filename = 'answers_{}.txt'.format(time.strftime("%d%b%Y_%H%M%S", time.localtime()))             
    output_file = open(os.path.join(output_dir, output_filename), 'w')

    for instance_filename in sorted(os.listdir(input_dir)):
        # importer l'instance depuis le fichier (attention code non robuste)
        instance_file = open(os.path.join(input_dir, instance_filename), "r")
        lines = instance_file.readlines()
        
        cout_opt = int(lines[1])
        k = int(lines[4])
        
        idx_demande = lines.index('# demandes\n')
        sites = {}
        for i in range(7,idx_demande-1):
            coords = lines[i].split()
            sites[i-7] = (int(coords[0]), int(coords[1]))

        demandes = [int(d) for d in lines[idx_demande+1].split()]
                
        # lancement de l'algo offline
        reponses = offline_k_serveurs(sites, k, demandes)

        # calcul cout
        cout_offline = 0
        serveurs = {i:(0,0) for i in range(k)} #tous les serveurs à la position 0,0 au départ
        for d,r in zip(demandes,reponses):
            cout_offline += manhatan_distance(serveurs[r], sites[d])
            serveurs[r] = sites[d]

        # ajout au rapport
        output_file.write(instance_filename)
        if cout_opt != cout_offline:
            output_file.write(': KO expected: {}, found: {}\n'.format(cout_opt, cout_offline))
        else:
            output_file.write(': OK\n')
        
    output_file.close()
