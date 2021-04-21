
import argparse
import socket
import struct
import math  
import time
import copy
import random

class Game():
    def __init__(self):
    	#Defining host and port
        host = "localhost"
        port = "5555" 

        #Parser definition
        parser = argparse.ArgumentParser()
        parser.add_argument("host")
        parser.add_argument("port")

        args = parser.parse_args()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, int(port)))

        # NME: Naming our Monster
        self.sock.send("NME".encode("ascii"))
        self.sock.send(struct.pack("1B", 8))
        self.sock.send("DeepBlue".encode("ascii"))

        # SET: Getting number of rows and columns of the map
        header = self.sock.recv(3).decode("ascii")
        if header != "SET":
            print("Protocol Error at SET")
        else:
            (self.height, self.width)= self.receive_data(self.sock, 2, "2B")

        # HUM: Getting information abour humans
        header = self.sock.recv(3).decode("ascii")
        if header != "HUM":
            print("Protocol Error at HUM")
        else:
            self.number_of_homes = self.receive_data(self.sock, 1, "1B")[0]
            self.homes_raw = self.receive_data(self.sock, self.number_of_homes * 2, "{}B".format(self.number_of_homes * 2))
            
        # HME: Getting initial position coordinates
        header = self.sock.recv(3).decode("ascii")
        if header != "HME":
            print("Protocol Error at HME")
        else:
            self.start_position = tuple(self.receive_data(self.sock, 2, "2B"))

        # MAP: Getting initial state of the map
        header = self.sock.recv(3).decode("ascii")
        if header != "MAP":
            print("Protocol Error at MAP")
        else:
            self.number_map_commands = self.receive_data(self.sock,1, "1B")[0]
            self.map_commands_raw = self.receive_data(self.sock, self.number_map_commands * 5, "{}B".format(self.number_map_commands * 5))

        #We create a board
        self.create_board()
        self.max_depth = 1 #Max Depth of the MiniMaxAlphaBeta Algorithm

        #Determining the board table index position following the order: (Nbr Humans, Nbr Vampires, Nbr Werewolves) 
        #and based on whether I am Vampire(,X,) or a Werewolf(,,X)
        if self.start_position == (4,3): #Vampire
            self.player_pos_array = 1
            self.enemy_pos_array = 2
        elif self.start_position == (4,1): #Werewolf
            self.player_pos_array = 2
            self.enemy_pos_array = 1
        else: #In case run on a different map, it will be Player 1
            self.player_pos_array = 1
            self.enemy_pos_array = 2        	

    def receive_data(self, sock, size, fmt):
        """
        Receiving data in bytes from the socket, unpacking and returning them
        """
        data = bytes()
        while len(data) < size:
            data += sock.recv(size - len(data))
        return struct.unpack(fmt, data)

    def create_board(self):
        """
        Creating the board based on height and width and filling it with the initial state of vampires, werewolves and humans
        """
        self.board_challenge = []
        for i in range(self.height):
            self.board_challenge.append([(0,0,0)]*self.width)

        #Creating a list of subtuples of 5 elements
        self.initial_positions_board = list(self.map_commands_raw[x:x + 5]  for x in range(0, len(self.map_commands_raw), 5))

        for i in range(self.height):
          for j in range(self.width):
            for k in range(0,len(self.initial_positions_board)): #Iterate through all initial positions
                if self.initial_positions_board[k][1]==i and self.initial_positions_board[k][0]==j:
                  self.board_challenge[i][j]=self.initial_positions_board[k][2:] #Filling in start positions in the board


    def calculateDistance(self,x1,y1,x2,y2):  
        """
        Compute Euclidean Distance in 2D of 2 given coordinates (x1,y1) and (x2,y2)
        """        
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
        return dist  

    def calculateDistance_plays(self,x_init,y_init,x_target,y_target):
        """
        Calculates the distance in number of plays between 2 given coordinates (x_init,y_init) and (x_target,y_target)
        """

        nbr_plays = 0 #Initializing number of plays

        #While we do not reach the target coordinates, we keep moving and counting
        while x_target != x_init or y_target != y_init:

            #Checking all possible moves
            next_poss_moves = self.possible_nxt_moves(x_init, y_init)
            #Splitting List into sublists of 2 elements
            next_moves_xy = tuple(next_poss_moves[x:x + 2]  for x in range(0, len(next_poss_moves), 2))

            dist_next_final = 100
            x_next_final = 0
            y_next_final = 0

            #Assigning next move based on closest distance to target
            for i in range(0,len(next_moves_xy)):

                x_next = next_moves_xy[i][0]
                y_next = next_moves_xy[i][1]

                distance_next = self.calculateDistance(x_next, y_next, x_target, y_target)
                if distance_next < dist_next_final: #If we find a closer distance
                    dist_next_final = distance_next
                    x_next_final = x_next
                    y_next_final = y_next

            nbr_plays += 1 #Counting
            #Updating coordinates
            x_init = x_next_final
            y_init = y_next_final
     
        return nbr_plays  

    def valid_coordinates(self,x,y):
        """
        Checking whether or not a given coordinates (x,y) are valid given a height and width of the map
        """          
        if (x >= 0 and x < self.height) and (y>=0 and y < self.width):
            return True
        else:
            return False

    def possible_nxt_moves(self,x,y):
        """
        Given a pair of coordinates (x,y) it computes all possible next moves
        """  
        nxt_moves = [] #Initializing possible next moves

        x_new = x 
        y_new = y + 1
        if self.valid_coordinates(x_new,y_new):
            nxt_moves.append(x_new)
            nxt_moves.append(y_new) 

        x_new = x + 1
        y_new = y + 1
        if self.valid_coordinates(x_new,y_new):
            nxt_moves.append(x_new)
            nxt_moves.append(y_new)

        x_new = x + 1
        y_new = y
        if self.valid_coordinates(x_new,y_new):
            nxt_moves.append(x_new)
            nxt_moves.append(y_new) 

        x_new = x + 1
        y_new = y - 1
        if self.valid_coordinates(x_new,y_new):
            nxt_moves.append(x_new)
            nxt_moves.append(y_new)

        x_new = x 
        y_new = y - 1
        if self.valid_coordinates(x_new,y_new):
            nxt_moves.append(x_new)
            nxt_moves.append(y_new)        

        x_new = x - 1
        y_new = y - 1
        if self.valid_coordinates(x_new,y_new):
            nxt_moves.append(x_new)
            nxt_moves.append(y_new)

        x_new = x - 1
        y_new = y
        if self.valid_coordinates(x_new,y_new):
            nxt_moves.append(x_new)
            nxt_moves.append(y_new)   

        x_new = x - 1
        y_new = y + 1
        if self.valid_coordinates(x_new,y_new):
            nxt_moves.append(x_new)
            nxt_moves.append(y_new)

        return nxt_moves

    def move_ai_single(self,x_ori,y_ori,nbr,x_des,y_des):
        """
        Moves one group of creatures of size nbr from an origin (x_ori,y_ori) to a target (x_des,y_des)
        Note that the server has the x-y coordinate system switched compared to our logic
        """         
        # MOV
        self.sock.send("MOV".encode("ascii"))
        self.sock.send(struct.pack("1B",1))
        self.sock.send(struct.pack("1B",x_ori))
        self.sock.send(struct.pack("1B",y_ori))
        self.sock.send(struct.pack("1B",nbr))
        self.sock.send(struct.pack("1B",x_des))
        self.sock.send(struct.pack("1B",y_des))

        
    def move_ai(self, action_source, action_target):
        """
        Moves many groups of creatures from action_source to a action_target
        Note that the server has the x-y coordinate system switched compared to our logic
        """  
        nbr_moves = 0 #Initializing number of moves
        for i in range(len(action_target)):
            nbr_moves += len(action_target[i]) #Updating total number of moves based on action target

        #Accounting for the case in which a monster remains on the same position (1 move less)
        for i in range(len(action_source)):
            for j in range(len(action_target[i])):
                if action_source[i] == action_target[i][j]:
                    nbr_moves -= 1
        
        # MOV: We move all creatures based on source and target data
        self.sock.send("MOV".encode("ascii"))
        self.sock.send(struct.pack("1B",nbr_moves))
        for i in range(len(action_source)):
            for j in range(len(action_target[i])):
                if action_source[i] == action_target[i][j]: #Continue for the case in which a monster remains on the same position
                    continue
                self.sock.send(struct.pack("1B",action_source[i][1]))
                self.sock.send(struct.pack("1B",action_source[i][0]))
                self.sock.send(struct.pack("1B",action_target[i][j][2]))
                self.sock.send(struct.pack("1B",action_target[i][j][1]))
                self.sock.send(struct.pack("1B",action_target[i][j][0]))
             
    def update_board(self):
        """
        Updates the board based on last move
        """          
        # UPD: Read last modifications of the map
        header = self.sock.recv(3).decode("ascii")
        if header != "UPD":
            print("Protocol Error at UPD")
        else:
            self.number_of_updates = self.receive_data(self.sock, 1, "1B")[0]
            self.updates = self.receive_data(self.sock, self.number_of_updates * 5, "{}B".format(self.number_of_updates * 5))

        #Creating a list of lists of 5 elements
        self.new_positions_map = list(self.updates[x:x + 5]  for x in range(0, len(self.updates), 5))

        #Updating
        for i in range(self.height):
          for j in range(self.width):
            for k in range(0,len(self.new_positions_map)): #Iterate through all board changes
                if self.new_positions_map[k][1]==i and self.new_positions_map[k][0]==j:
                  self.board_challenge[i][j]=self.new_positions_map[k][2:] #Update board with new values

    def print_board(self):
        """
        Prints current state of the board
        """             
        for i in range(self.height):
            print(self.board_challenge[i])

    def is_terminal(self, board):
        """
        Checks if the game has finished
        The game finishes when one of the creatures disappears
        or when there is only 1 group of each creature in the map and both are in the same position
        """         
        pos_monsters = []
        pos_monsters_enemy = []

        #Reading monster and enemy positions
        for i in range(self.height):
          for j in range(self.width):
            if board[i][j][self.player_pos_array]>0:
                pos_monsters.append((i,j))
            elif board[i][j][self.enemy_pos_array]>0:
                pos_monsters_enemy.append((i,j))

        if len(pos_monsters) == 1 and len(pos_monsters_enemy) == 1: #If there is only 1 group of each creature in the map and both are in the same position
            if pos_monsters[0] == pos_monsters_enemy[0]:
                return True
        elif len(pos_monsters) == 0 or len(pos_monsters_enemy) == 0: #If one of the creatures has disappeared
            return True

        return False

    def possible_nxt_moves_split(self,x,y,number,len_monsters):
        """
        Given a pair of coordinates (x,y) it computes all possible next moves considering splis as well      
        """           
        poss_moves = self.possible_nxt_moves(x,y) #Reading adjacent cells

        #When we have more than 1 group of monsters of the same type, it is possible to keep one group in the same position
        if len_monsters>1:
            poss_moves.append(x)
            poss_moves.append(y)

        #List of next moves that do not involve a split
        nxt_moves_nosplit_prev = list(poss_moves[x:x + 2] + [number] for x in range(0, len(poss_moves), 2))
        nxt_moves_nosplit = list(nxt_moves_nosplit_prev[x:x + 1] for x in range(0, len(nxt_moves_nosplit_prev)))        

        poss_moves_split = []
        poss_moves_xy = list(poss_moves[x:x + 2]  for x in range(0, len(poss_moves), 2))   

        if number == 4:
            for i in range(len(poss_moves_xy)):
                for j in range(i+1,len(poss_moves_xy)):
                    poss_moves_split.append(poss_moves_xy[i][0])
                    poss_moves_split.append(poss_moves_xy[i][1])
                    poss_moves_split.append(2)          
                    poss_moves_split.append(poss_moves_xy[j][0])
                    poss_moves_split.append(poss_moves_xy[j][1])
                    poss_moves_split.append(2)

        #List of next moves that involve a split
        nxt_moves_split_prev = list(poss_moves_split[x:x + 3]  for x in range(0, len(poss_moves_split), 3))
        nxt_moves_split = list(nxt_moves_split_prev[x:x + 2]  for x in range(0, len(nxt_moves_split_prev), 2))

        #We only split our monsters in maximum 2 groups
        if len_monsters==1:
            #Merging both next move lists (split and no split)
            nxt_moves = nxt_moves_nosplit + nxt_moves_split
        else:
            nxt_moves = nxt_moves_nosplit

        return nxt_moves

    def eval_function(self,board):
        """
        Function that computes the Heuristics for the current state of the board     
        """     

        pos_humans = []
        nbr_humans = []
        pos_monsters = []
        nbr_monsters = []
        pos_monsters_enemy = []
        nbr_monsters_enemy = [] 

        nbr_monsters_tot = 0    
        nbr_monsters_enemy_tot = 0   

        #Reading all positions of the current board
        for i in range(self.height):
          for j in range(self.width):
            if board[i][j][self.player_pos_array]>0:
                pos_monsters.append((i,j))
                nbr_monsters.append(board[i][j][self.player_pos_array]) 
                nbr_monsters_tot += board[i][j][self.player_pos_array]
                # x = i
                # y = j
                # nbr_monsters = board[i][j][self.player_pos_array]   
            if board[i][j][self.enemy_pos_array]>0:
                pos_monsters_enemy.append((i,j))
                nbr_monsters_enemy.append(board[i][j][self.enemy_pos_array])
                nbr_monsters_enemy_tot += board[i][j][self.enemy_pos_array]
                # x_enemy = i
                # y_enemy = j
                # nbr_monsters_enemy = board[i][j][self.enemy_pos_array]
            if board[i][j][0]>0:
                pos_humans.append((i,j))
                nbr_humans.append(board[i][j][0])

        ###EVALUATION ENEMY###
        #Heuristic based on the position and number of creatures of the enemy 
        #This heuristic is evaluated for each subgroup of our monsters
        weight_to_enemy = [] #weight for the closest enemy to the monster
        monster_enemy_diff = [] #difference between the number of monsters on each position and that of the closest group of enemies

        for i in range(len(pos_monsters)):
            distance_enemy_min = 100
            nbr_enemy_min=nbr_monsters[i] #Initializing to make the difference 0 (so that it does not affect the evaluation) in case there are no more enemies
            for j in range(len(pos_monsters_enemy)):
                distance_enemy = self.calculateDistance_plays(pos_monsters[i][0],pos_monsters[i][1],pos_monsters_enemy[j][0],pos_monsters_enemy[j][1])
                if distance_enemy < distance_enemy_min:
                    distance_enemy_min = distance_enemy
                    nbr_enemy_min = nbr_monsters_enemy[j]

            if distance_enemy_min == 0: #When distance 0, we already evaluated it in the final heuristic formula
                weight_to_enemy.append(0) 
            elif distance_enemy_min == 1:
                weight_to_enemy.append(0.9)
            elif distance_enemy_min > 1: #For distance 2,3,4... we disregard the enemy evaluation 
                weight_to_enemy.append(0)           

            #Computing probability based on monster numbers
            if nbr_monsters[i]>1.5*nbr_enemy_min: #If 1.5 times greater
                prob=1
            elif nbr_enemy_min>1.5*nbr_monsters[i]:
                prob=0
            elif nbr_monsters[i]>nbr_enemy_min:
                prob=(nbr_monsters[i]/nbr_enemy_min)-0.5
            elif nbr_monsters[i]==nbr_enemy_min:
                prob=0.5
            elif nbr_monsters[i]<nbr_enemy_min:
                prob=nbr_monsters[i]/(2*nbr_enemy_min)

            #Analyse probability to determine whether the attacking monster wins or loses                                  
            nbr_monsters_future = nbr_monsters[i]*prob*prob
            nbr_enemy_future = nbr_enemy_min*(1-prob)*(1-prob)

            diff_init = nbr_monsters[i] - nbr_enemy_min
            diff_future = nbr_monsters_future - nbr_enemy_future
            diff = diff_future - diff_init
            monster_enemy_diff.append(diff)

        eval_enemy = 0
        #Evaluating based on distance from all group of monsters to the closest enemy's
        for i in range(len(pos_monsters)):
            eval_enemy = eval_enemy + weight_to_enemy[i]*monster_enemy_diff[i]

        ###EVALUATION DIFFERENCE MONSTER VS ENEMY FUTURE TOTAL###
        #Heuristic based on the potential fight between our monsters and the enemy's 
        nbr_enemy_tot_temp = nbr_monsters_enemy_tot
        nbr_monsters_future_accum = 0

        for i in range(len(pos_monsters)):

            if nbr_monsters[i]>1.5*nbr_enemy_tot_temp: #If 1.5 times greater then all enemy monsters killed
                prob=1
            elif nbr_enemy_tot_temp>1.5*nbr_monsters[i]:
                prob=0
            elif nbr_monsters[i]>nbr_enemy_tot_temp:
                prob=(nbr_monsters[i]/nbr_enemy_tot_temp)-0.5
            elif nbr_monsters[i]==nbr_enemy_tot_temp:
                prob=0.5
            elif nbr_monsters[i]<nbr_enemy_tot_temp:
                prob=nbr_monsters[i]/(2*nbr_enemy_tot_temp)

            nbr_monsters_future = nbr_monsters[i]*prob*prob
            nbr_monsters_future_accum += nbr_monsters_future
            nbr_enemy_tot_temp = nbr_enemy_tot_temp*(1-prob)*(1-prob)

        diff_future_total = nbr_monsters_future_accum - nbr_enemy_tot_temp

        ###EVALUATION WEIGHT (DISTANCE) BETWEEN SPLITS OF THE SAME MONSTER###
        #Heuristic based on the distance between subgroups of the same type of monster
        if len(pos_monsters) > 1:
        #if len(pos_monsters) > 1 and len(pos_humans) == 0:
            dist_between_monsters = self.calculateDistance_plays(pos_monsters[0][0],pos_monsters[0][1],pos_monsters[1][0],pos_monsters[1][1])
            weight_between_monsters = -0.02*dist_between_monsters + 0.22 # (1, 0.20); (2, 0.18); (3, 0.16); (4, 0.14) ...
        else:
            weight_between_monsters = 0

        ###EVALUATION MONSTER TO HUMANS###
        #Heuristic based on the distance and number of monsters with respect to the closest human's
        #This heuristic is evaluated for each subgroup of our monsters
        
        #Sorting in descending order based on number of humans in order to start anlyzing the biggest number of humans at each turn
        pos_humans_sorted = [x for _,x in sorted(zip(nbr_humans,pos_humans),reverse=True)] 
        nbr_humans_sorted = sorted(nbr_humans,reverse=True)

        pos_humans_closer = [] #Array containing the position of the humans closer to our monsters compared to the enemy
        nbr_humans_closer = [] #Array containing the number of humans of a particular position closer to our monsters

        pos_humans_not_closer = [] #Array containing the position of the humans not closer to our monsters compared to the enemy
        nbr_humans_not_closer = [] #Array containing the number of humans of a particular position not closer to our monsters

        #Computing the humans closer to monster
        for i in range(len(pos_humans_sorted)):
            dist_min_to_monster = 100
            for j in range(len(pos_monsters)):
                distance_to_monster = self.calculateDistance_plays(pos_monsters[j][0],pos_monsters[j][1],pos_humans_sorted[i][0],pos_humans_sorted[i][1])
                if distance_to_monster < dist_min_to_monster:
                    dist_min_to_monster = distance_to_monster

            dist_min_to_enemy = 100
            for j in range(len(pos_monsters_enemy)):
                distance_to_enemy = self.calculateDistance_plays(pos_monsters_enemy[j][0],pos_monsters_enemy[j][1],pos_humans_sorted[i][0],pos_humans_sorted[i][1])
                if distance_to_enemy < dist_min_to_enemy:
                    dist_min_to_enemy = distance_to_enemy

            if dist_min_to_monster < dist_min_to_enemy:
                pos_humans_closer.append(pos_humans_sorted[i])
                nbr_humans_closer.append(nbr_humans_sorted[i])
            else:
                pos_humans_not_closer.append(pos_humans_sorted[i])
                nbr_humans_not_closer.append(nbr_humans_sorted[i])                

        #Making a copy to edit them
        pos_humans_closer_cp = copy.deepcopy(pos_humans_closer)
        nbr_humans_closer_cp = copy.deepcopy(nbr_humans_closer)
        pos_humans_not_closer_cp = copy.deepcopy(pos_humans_not_closer)
        nbr_humans_not_closer_cp = copy.deepcopy(nbr_humans_not_closer)

        monst_to_human_target = [] #Array that contains the potential number of humans converted
        weight_monst_to_human_target = [] #Array that contains a coefficient based on the distance to the group of humans to be converted       

        #Computing the closest human to each monster. If more than 1 group of humans closest to monster, it selects the largest human group
        for i in range(len(pos_monsters)):
            #If there are humans closer to monster left for the analysis
            if len(pos_humans_closer_cp)>0:
                distance_to_human_min = 100
                for j in range(len(pos_humans_closer_cp)):
                    distance_to_human = self.calculateDistance_plays(pos_monsters[i][0],pos_monsters[i][1],pos_humans_closer_cp[j][0],pos_humans_closer_cp[j][1])
                    if distance_to_human < distance_to_human_min:
                        distance_to_human_min = distance_to_human     
                        idx_human = j    

                #Computing probability based on monster numbers
                if nbr_monsters[i]>=nbr_humans_closer_cp[idx_human]: #If monsters are at least as numerous as humans they convert them
                    prob=1
                else: #If not, random battle starts
                    prob=nbr_monsters[i]/(2*nbr_humans_closer_cp[idx_human])

                 #Analyse probability to determine whether the attacking monster wins or loses
                if prob > random.random(): #If attacking monster wins
                    future_nbr_monster = nbr_monsters[i]*prob+nbr_humans_closer_cp[idx_human]*prob
                    gain = future_nbr_monster - nbr_monsters[i]
                else: #if attacking monster loses
                    future_nbr_monster = 0
                    gain = -nbr_monsters[i]

                monst_to_human_target.append(gain) #Save the potential gain
                funct_dist_hum = -0.02*distance_to_human_min + 0.22 # (1, 0.20); (2, 0.18); (3, 0.16); (4, 0.14) ...
                weight_monst_to_human_target.append(funct_dist_hum) #Save the distance coefficient
                #Pop Human so that next monster does not see it again
                pos_humans_closer_cp.pop(idx_human)
                nbr_humans_closer_cp.pop(idx_human)

            #If there are humans not closer to monster left for the analysis
            elif len(pos_humans_not_closer_cp)>0:
                distance_to_human_min = 100
                for j in range(len(pos_humans_not_closer_cp)):
                    distance_to_human = self.calculateDistance_plays(pos_monsters[i][0],pos_monsters[i][1],pos_humans_not_closer_cp[j][0],pos_humans_not_closer_cp[j][1])
                    if distance_to_human < distance_to_human_min:
                        distance_to_human_min = distance_to_human     
                        idx_human = j    

                #Computing probability based on monster numbers
                if nbr_monsters[i]>=nbr_humans_not_closer_cp[idx_human]: #If monsters are at least as numerous as humans they convert them
                    prob=1
                else: #If not, random battle starts
                    prob=nbr_monsters[i]/(2*nbr_humans_not_closer_cp[idx_human])

                 #Analyse probability to determine whether the attacking monster wins or loses
                if prob > random.random(): #If attacking monster wins
                    future_nbr_monster = nbr_monsters[i]*prob+nbr_humans_not_closer_cp[idx_human]*prob
                    gain = future_nbr_monster - nbr_monsters[i]
                else: #if attacking monster loses
                    future_nbr_monster = 0
                    gain = -nbr_monsters[i]

                monst_to_human_target.append(gain) #Save the potential gain
                funct_dist_hum = -0.02*distance_to_human_min + 0.22 # (1, 0.20); (2, 0.18); (3, 0.16); (4, 0.14) ...
                weight_monst_to_human_target.append(funct_dist_hum) #Save the distance coefficient
                #Pop Human so that next monster doesnt see it again
                pos_humans_not_closer_cp.pop(idx_human)
                nbr_humans_not_closer_cp.pop(idx_human)

        eval_monster_to_humans = 0

        for i in range(len(monst_to_human_target)):
            eval_monster_to_humans = eval_monster_to_humans + weight_monst_to_human_target[i]*monst_to_human_target[i]
     
        ###EVALUATION ALL HUMANS###
        #Heuristic based on the distance and number of monsters with respect to all humans
        #This heuristic is evaluated for each group of humans    
            
        evaluation_humans = 0
        
        if len(pos_monsters) > 0: #If monsters left for the analysis   
             
            nbr_humans_target = [] #array for the potential number of humans converted (gain)
            weight_humans_target = [] #array for the weight associated with the potential number of humans converted (gain)

            #Iterate through all group of humans
            for i in range(len(pos_humans)):
                #Computing closest monster to a given human
                dist_monst_humans_min = 100
                for j in range(len(pos_monsters)):
                    dist_monst_humans = self.calculateDistance_plays(pos_monsters[j][0],pos_monsters[j][1],pos_humans[i][0],pos_humans[i][1])
                    if dist_monst_humans < dist_monst_humans_min:
                        dist_monst_humans_min = dist_monst_humans
                        idx_monster = j

                #Computing closest enemy to a given human
                dist_enemy_humans_min = 100
                for j in range(len(pos_monsters_enemy)):
                    dist_enemy_humans = self.calculateDistance_plays(pos_monsters_enemy[j][0],pos_monsters_enemy[j][1],pos_humans[i][0],pos_humans[i][1])
                    if dist_enemy_humans < dist_enemy_humans_min:
                        dist_enemy_humans_min = dist_enemy_humans

                #Computes difference of distances of closest monster and closest enemy to a given human
                diff_distance = dist_monst_humans_min - dist_enemy_humans_min 

                #Computing probability based on monster numbers
                if nbr_monsters[idx_monster]>=nbr_humans[i]: #If monsters are at least as numerous as humans they convert them
                    prob=1
                else: #If not, random battle starts
                    prob=nbr_monsters[idx_monster]/(2*nbr_humans[i])

                 #Analyse probability to determine whether the attacking monster wins or loses
                if prob > random.random(): #If attacking monster wins
                    future_nbr_monster = nbr_monsters[idx_monster]*prob+nbr_humans[i]*prob
                    gain = future_nbr_monster - nbr_monsters[idx_monster]
                else: #if attacking monster loses
                    future_nbr_monster = 0
                    gain = -nbr_monsters[idx_monster]

                nbr_humans_target.append(gain) #Save the potential gain

                if diff_distance < 0: #If I am closer to the human group than the enemy
                    weight_humans_target.append(0.75)
                elif diff_distance == 0: #If I am at the same distance to the human group than the enemy
                    weight_humans_target.append(0.2)
                elif diff_distance > 0: #If I am further to the human group than the enemy
                    weight_humans_target.append(0)
    
            #Evaluating based on number and distance to humans in map    
            for i in range(len(pos_humans)):
                evaluation_humans = evaluation_humans + weight_humans_target[i]*nbr_humans_target[i]

        ###FINAL HEURISTIC###
        evaluation = 1*evaluation_humans + 1*eval_monster_to_humans + 1.5*(nbr_monsters_tot - nbr_monsters_enemy_tot) + \
                        0.1*weight_between_monsters + 0.01*diff_future_total + 0.2*eval_enemy

        return evaluation

    def minimax_alphabeta(self, current_depth, board, is_max_turn, alpha, beta):
        """
        Function that implements the minimax alphabeta algorithm    
        current_depth: current depth of the analysis in which we are in
        board: board to analyze
        is_max_turn: True or False to determine whether we are analyzing player 1 or player 2 (enemy) 
        alpha: variable for player 1
        beta: variable for player 2

        We first update the board with a next possible move of a player, update the parameters (current_depth, is_max_turn, alpha, beta) 
        and then call again this function until we reach the maximum depth or the end of the game
        """  

        #If we are in the maximum depth allowed for the algorithm or if the game has ended, we want to compute our Heuristic
        if current_depth == self.max_depth or self.is_terminal(board):
            return self.eval_function(board), [], []

        pos_humans = []
        nbr_humans = []
        pos_monsters = []
        nbr_monsters = []
        pos_monsters_enemy = []
        nbr_monsters_enemy = []

        #Reading information about our monsters, the enemy and the humans in the board
        for i in range(self.height):
          for j in range(self.width):
            if board[i][j][0]>0:
                pos_humans.append((i,j))
                nbr_humans.append(board[i][j][0])

            if is_max_turn == True:
                if board[i][j][self.player_pos_array]>0:
                    pos_monsters.append((i,j))
                    nbr_monsters.append(board[i][j][self.player_pos_array])
                elif board[i][j][self.enemy_pos_array]>0:
                    pos_monsters_enemy.append((i,j))
                    nbr_monsters_enemy.append(board[i][j][self.enemy_pos_array])                    
            else:
                if board[i][j][self.enemy_pos_array]>0:
                    pos_monsters.append((i,j))
                    nbr_monsters.append(board[i][j][self.enemy_pos_array])                    
                elif board[i][j][self.player_pos_array]>0:
                    pos_monsters_enemy.append((i,j))
                    nbr_monsters_enemy.append(board[i][j][self.player_pos_array])                        
        
        #Initializing our source, target vectors and the best value heuristics
        action_source = [0]*len(pos_monsters)
        action_target = [0]*len(pos_monsters)
        best_value = 0

        ###############LOGIC WHEN GROUP OF MONSTERS = 1#########################

        if len(pos_monsters) == 1:

            #Iterate through all monsters
            for monst in range(len(pos_monsters)):
                x = pos_monsters[monst][0]
                y = pos_monsters[monst][1]
                number = nbr_monsters[monst]

                next_moves_xy = self.possible_nxt_moves_split(x,y,number,len(pos_monsters))
                best_value = float('-inf') if is_max_turn else float('inf') #Initialize Best Value
                random.shuffle(next_moves_xy) #Shuffle list of possible next moves

                #Iterate through all possible next moves
                for k in range(len(next_moves_xy)):
                    new_board = copy.deepcopy(board) #Create a copy of the board to edit it
                    #Within a particular next move, iterate in case there are splits
                    for l in range(len(next_moves_xy[k])):

                        x_next = next_moves_xy[k][l][0]
                        y_next = next_moves_xy[k][l][1]   
                        number_next = next_moves_xy[k][l][2]         
                        flag_fight_humans = 0 #Flag to determine if there are humans in the next position
                        flag_fight_enemy = 0 #Flag to determine if there are enemies in the next position

                        if is_max_turn == True:
                            #Checking if next move corresponds to a human's position
                            for h in range(len(pos_humans)):
                                if pos_humans[h][0]==x_next and pos_humans[h][1]==y_next:
                                    flag_fight_humans = 1

                                    #Computing probability based on monster numbers
                                    if number_next>=nbr_humans[h]: #If monsters are at least as numerous as humans they convert them
                                        prob=1
                                    else: #If not, random battle starts
                                        prob=number_next/(2*nbr_humans[h])

                                    #Analyse probability to determine whether the attacking monster wins or loses
                                    if prob > random.random(): #If attacking monster wins
                                        new_board[x_next][y_next][self.player_pos_array]=number_next*prob+nbr_humans[h]*prob
                                        new_board[x_next][y_next][0] = 0
                                        new_board[x][y][self.player_pos_array]-=number_next
                                    else: #if attacking monster loses
                                        new_board[x_next][y_next][0] = nbr_humans[h]*(1-prob)
                                        new_board[x][y][self.player_pos_array]-=number_next    

                            #Checking if next move corresponds to an enemy's position
                            for e in range(len(pos_monsters_enemy)):
                                if pos_monsters_enemy[e][0]==x_next and pos_monsters_enemy[e][1]==y_next:
                                    flag_fight_enemy = 1

                                    #Computing probability based on monster numbers
                                    if number_next>1.5*nbr_monsters_enemy[e]: #If 1.5 times greater then all enemy monsters killed
                                        prob=1
                                    elif nbr_monsters_enemy[e]>1.5*number_next:
                                        prob=0
                                    elif number_next>nbr_monsters_enemy[e]:
                                        prob=(number_next/nbr_monsters_enemy[e])-0.5
                                    elif number_next==nbr_monsters_enemy[e]:
                                        prob=0.5
                                    elif number_next<nbr_monsters_enemy[e]:
                                        prob=number_next/(2*nbr_monsters_enemy[e])

                                    #Analyse probability to determine whether the attacking monster wins or loses                                  
                                    new_board[x_next][y_next][self.player_pos_array]=number_next*prob*prob
                                    new_board[x_next][y_next][self.enemy_pos_array] = nbr_monsters_enemy[e]*(1-prob)*(1-prob)
                                    new_board[x][y][self.player_pos_array]-=number_next

                            #If neither humans nor enemy monsters were in the next position then we just move our attacking monsters entirely
                            if flag_fight_humans == 0 and flag_fight_enemy == 0:
                                new_board[x_next][y_next][self.player_pos_array]+=number_next
                                new_board[x][y][self.player_pos_array]-=number_next
                        else: #Same for is_max_turn == False
                            #Checking if next move corresponds to a human's position
                            for h in range(len(pos_humans)):
                                if pos_humans[h][0]==x_next and pos_humans[h][1]==y_next:
                                    flag_fight_humans = 1

                                    #Computing probability based on monster numbers
                                    if number_next>=nbr_humans[h]: #If monsters are at least as numerous as humans they convert them
                                        prob=1
                                    else: #If not, random battle starts
                                        prob=number_next/(2*nbr_humans[h])

                                    #Analyse probability to determine whether the attacking monster wins or loses
                                    if prob > random.random(): #If attacking monster wins
                                        new_board[x_next][y_next][self.enemy_pos_array]=number_next*prob+nbr_humans[h]*prob
                                        new_board[x_next][y_next][0] = 0
                                        new_board[x][y][self.enemy_pos_array]-=number_next
                                    else: #if attacking monster loses
                                        new_board[x_next][y_next][0] = nbr_humans[h]*(1-prob)
                                        new_board[x][y][self.enemy_pos_array]-=number_next    

                            #Checking if next move corresponds to an enemy's position
                            for e in range(len(pos_monsters_enemy)):
                                if pos_monsters_enemy[e][0]==x_next and pos_monsters_enemy[e][1]==y_next:
                                    flag_fight_enemy = 1

                                    #Computing probability based on monster numbers
                                    if number_next>1.5*nbr_monsters_enemy[e]: #If 1.5 times greater then all enemy monsters killed
                                        prob=1
                                    elif nbr_monsters_enemy[e]>1.5*number_next:
                                        prob=0
                                    elif number_next>nbr_monsters_enemy[e]:
                                        prob=(number_next/nbr_monsters_enemy[e])-0.5
                                    elif number_next==nbr_monsters_enemy[e]:
                                        prob=0.5
                                    elif number_next<nbr_monsters_enemy[e]:
                                        prob=number_next/(2*nbr_monsters_enemy[e])

                                    #Analyse probability to determine whether the attacking monster wins or loses                                   
                                    new_board[x_next][y_next][self.enemy_pos_array]=number_next*prob*prob
                                    new_board[x_next][y_next][self.player_pos_array] = nbr_monsters_enemy[e]*(1-prob)*(1-prob)
                                    new_board[x][y][self.enemy_pos_array]-=number_next

                            #If neither humans nor enemy monsters were in the next position then we just move our attacking monsters entirely
                            if flag_fight_humans == 0 and flag_fight_enemy == 0:
                                new_board[x_next][y_next][self.enemy_pos_array]+=number_next
                                new_board[x][y][self.enemy_pos_array]-=number_next

                    #Call minimax alphabeta again with new updated board, with 1 depth in addition and changing to the other player (not is_max_turn)
                    eval_child, source_child, target_child = self.minimax_alphabeta(current_depth+1,new_board,not is_max_turn, alpha, beta)

                    #We want to display the heuristic at the end of the analysis
                    if current_depth==0:
                        print("Evaluation for monster {}, for next move {}: {}".format(pos_monsters[monst],next_moves_xy[k],eval_child))

                    #Applying minimax alphabeta pruning
                    if is_max_turn and best_value < eval_child: #If our child evaluation is greater
                        best_value = eval_child #Update greatest evaluation
                        action_source[monst] = [x,y,number] #Save our best results for source
                        action_target[monst] = next_moves_xy[k] #Save our best results for target
                        alpha = max(alpha, best_value) #Update alpha
                        if beta <= alpha: #Pruning
                            break

                    elif (not is_max_turn) and best_value > eval_child: #If our child evaluation is smaller
                        best_value = eval_child #Update smallest evaluation
                        action_source[monst] = [x,y,number] #Save our best results for source                
                        action_target[monst] = next_moves_xy[k] #Save our best results for target
                        beta = min(beta, best_value) #Update beta
                        if beta <= alpha: #Pruning
                            break

        ###############LOGIC WHEN GROUP OF MONSTERS > 1#########################

        elif len(pos_monsters) > 1:
            next_moves_xy_total = [0]*len(pos_monsters)

            #Iterate through all monsters
            for monst in range(len(pos_monsters)):
                x = pos_monsters[monst][0]
                y = pos_monsters[monst][1]
                number = nbr_monsters[monst]

                next_moves_xy_monst = self.possible_nxt_moves_split(x,y,number,len(pos_monsters)) #Next possible moves
                random.shuffle(next_moves_xy_monst)
                next_moves_xy_total[monst] = next_moves_xy_monst

            best_value = float('-inf') if is_max_turn else float('inf') #Initialize Best Value

            #Iterate through all possible next moves
            for m1 in range(len(next_moves_xy_total[0])):
                break_activated = 0 #Variable to implement Alpha-Beta Pruning
                for m2 in range(len(next_moves_xy_total[1])):

                    #We cannot have all monsters stay in their same position. At least one has to move
                    #Neither can we let a cell to be target and source at the same time
                    #Also, we ignore the case in which 2 subgroup of monsters swap places because it does not provide any change overall
                    if (((pos_monsters[0][0] == next_moves_xy_total[0][m1][0][0] and pos_monsters[0][1] == next_moves_xy_total[0][m1][0][1]) 
                        and
                        (pos_monsters[1][0] == next_moves_xy_total[1][m2][0][0] and pos_monsters[1][1] == next_moves_xy_total[1][m2][0][1])) or

                        ((next_moves_xy_total[0][m1][0][0] == pos_monsters[1][0] and next_moves_xy_total[0][m1][0][1] == pos_monsters[1][1])
                        and 
                        (pos_monsters[1][0] != next_moves_xy_total[1][m2][0][0] or pos_monsters[1][1] != next_moves_xy_total[1][m2][0][1])) or

                        ((next_moves_xy_total[1][m2][0][0] == pos_monsters[0][0] and next_moves_xy_total[1][m2][0][1] == pos_monsters[0][1])
                        and
                        (pos_monsters[0][0] != next_moves_xy_total[0][m1][0][0] or pos_monsters[0][1] != next_moves_xy_total[0][m1][0][1])) or

                        ((next_moves_xy_total[0][m1][0][0] == pos_monsters[1][0] and next_moves_xy_total[0][m1][0][1] == pos_monsters[1][1])
                        and 
                        (next_moves_xy_total[1][m2][0][0] == pos_monsters[0][0] and next_moves_xy_total[1][m2][0][1] == pos_monsters[0][1]))):
                        
                        continue

                    new_board = copy.deepcopy(board)
                    pos_humans_cp = copy.deepcopy(pos_humans)
                    nbr_humans_cp = copy.deepcopy(nbr_humans)
                    pos_monsters_enemy_cp = copy.deepcopy(pos_monsters_enemy)
                    nbr_monsters_enemy_cp = copy.deepcopy(nbr_monsters_enemy)

                    #Update the board as many times as monsters we have
                    for monst in range(len(pos_monsters)):                                                   
                        x = pos_monsters[monst][0]
                        y = pos_monsters[monst][1]
                        number = nbr_monsters[monst]

                        #Creating index based on monster
                        if monst == 0:
                            idx = m1
                        elif monst == 1:
                            idx = m2

                        next_moves_xy = next_moves_xy_total[monst]

                        #Within a particular next move, iterate in case there are splits
                        for l in range(len(next_moves_xy[idx])):

                            x_next = next_moves_xy[idx][l][0]
                            y_next = next_moves_xy[idx][l][1]   
                            number_next = next_moves_xy[idx][l][2]         

                            flag_fight_humans = 0 #Flag to determine if there are humans in the next position
                            flag_fight_enemy = 0 #Flag to determine if there are enemies in the next position
                            if is_max_turn == True:
                                #Checking if next move corresponds to a human's position
                                for h in range(len(pos_humans_cp)):
                                    if pos_humans_cp[h][0]==x_next and pos_humans_cp[h][1]==y_next:
                                        flag_fight_humans = 1
                                        h_attacked = h

                                        #Computing probability based on monster numbers
                                        if number_next>=nbr_humans_cp[h]: #If monsters are at least as numerous as humans they convert them
                                            prob=1
                                        else: #If not, random battle starts
                                            prob=number_next/(2*nbr_humans_cp[h])

                                        #Analyse probability to determine whether the attacking monster wins or loses
                                        if prob > random.random(): #If attacking monster wins
                                            new_board[x_next][y_next][self.player_pos_array]=number_next*prob+nbr_humans_cp[h]*prob
                                            new_board[x_next][y_next][0] = 0
                                            new_board[x][y][self.player_pos_array]-=number_next
                                            h_eaten = 1
                                        else: #if attacking monster loses
                                            new_board[x_next][y_next][0] = nbr_humans_cp[h]*(1-prob)
                                            new_board[x][y][self.player_pos_array]-=number_next   
                                            h_eaten = 0 

                                #Updating human arrays based on board updates
                                if flag_fight_humans == 1:
                                    if h_eaten == 1:
                                        pos_humans_cp.pop(h_attacked)
                                        nbr_humans_cp.pop(h_attacked)

                                #Checking if next move corresponds to an enemy's position
                                for e in range(len(pos_monsters_enemy_cp)):
                                    if pos_monsters_enemy_cp[e][0]==x_next and pos_monsters_enemy_cp[e][1]==y_next:
                                        flag_fight_enemy = 1
                                        e_attacked = e

                                        #Computing probability based on monster numbers
                                        if number_next>1.5*nbr_monsters_enemy_cp[e]: #If 1.5 times greater then all enemy monsters killed
                                            prob=1
                                        if nbr_monsters_enemy_cp[e]>1.5*number_next:
                                            prob=0
                                        elif number_next>nbr_monsters_enemy_cp[e]:
                                            prob=(number_next/nbr_monsters_enemy_cp[e])-0.5
                                        elif number_next==nbr_monsters_enemy_cp[e]:
                                            prob=0.5
                                        elif number_next<nbr_monsters_enemy_cp[e]:
                                            prob=number_next/(2*nbr_monsters_enemy_cp[e])

                                        prob_attacked = prob
                                        #Analyse probability to determine whether the attacking monster wins or loses                                  
                                        new_board[x_next][y_next][self.player_pos_array]=number_next*prob*prob
                                        new_board[x_next][y_next][self.enemy_pos_array] = nbr_monsters_enemy_cp[e]*(1-prob)*(1-prob)
                                        new_board[x][y][self.player_pos_array]-=number_next

                                #Updating monster arrays based on board updates
                                if flag_fight_enemy == 1:
                                    future_monsters_enemy = (1-prob_attacked)*(1-prob_attacked)*nbr_monsters_enemy_cp[e_attacked]
                                    if future_monsters_enemy == 0:
                                        pos_monsters_enemy_cp.pop(e_attacked)
                                        nbr_monsters_enemy_cp.pop(e_attacked)
                                    else:
                                        nbr_monsters_enemy_cp[e_attacked] = future_monsters_enemy

                                #If neither humans nor enemy monsters were in the next position then we just move our attacking monsters entirely
                                if flag_fight_humans == 0 and flag_fight_enemy == 0:
                                    new_board[x_next][y_next][self.player_pos_array]+=number_next
                                    new_board[x][y][self.player_pos_array]-=number_next
                            else:
                                #Checking if next move corresponds to a human's position
                                for h in range(len(pos_humans_cp)):
                                    if pos_humans_cp[h][0]==x_next and pos_humans_cp[h][1]==y_next:
                                        flag_fight_humans = 1
                                        h_attacked = h

                                        #Computing probability based on monster numbers
                                        if number_next>=nbr_humans_cp[h]: #If monsters are at least as numerous as humans they convert them
                                            prob=1
                                        else: #If not, random battle starts
                                            prob=number_next/(2*nbr_humans_cp[h])

                                        #Analyse probability to determine whether the attacking monster wins or loses
                                        if prob > random.random(): #If attacking monster wins
                                            new_board[x_next][y_next][self.enemy_pos_array]=number_next*prob+nbr_humans_cp[h]*prob
                                            new_board[x_next][y_next][0] = 0
                                            new_board[x][y][self.enemy_pos_array]-=number_next
                                            h_eaten = 1
                                        else: #if attacking monster loses
                                            new_board[x_next][y_next][0] = nbr_humans_cp[h]*(1-prob)
                                            new_board[x][y][self.enemy_pos_array]-=number_next    
                                            h_eaten = 0
                                
                                #Updating human arrays based on board updates
                                if flag_fight_humans == 1:
                                    if h_eaten == 1:
                                        pos_humans_cp.pop(h_attacked)
                                        nbr_humans_cp.pop(h_attacked)

                                #Checking if next move corresponds to an enemy's position
                                for e in range(len(pos_monsters_enemy_cp)):
                                    if pos_monsters_enemy_cp[e][0]==x_next and pos_monsters_enemy_cp[e][1]==y_next:
                                        flag_fight_enemy = 1
                                        e_attacked = e

                                        #Computing probability based on monster numbers
                                        if number_next>1.5*nbr_monsters_enemy_cp[e]: #If 1.5 times greater then all enemy monsters killed
                                            prob=1
                                        elif nbr_monsters_enemy_cp[e]>1.5*number_next:
                                            prob=0
                                        elif number_next>nbr_monsters_enemy_cp[e]:
                                            prob=(number_next/nbr_monsters_enemy_cp[e])-0.5
                                        elif number_next==nbr_monsters_enemy_cp[e]:
                                            prob=0.5
                                        elif number_next<nbr_monsters_enemy_cp[e]:
                                            prob=number_next/(2*nbr_monsters_enemy_cp[e])

                                        prob_attacked = prob
                                        #Analyse probability to determine whether the attacking monster wins or loses                                   
                                        new_board[x_next][y_next][self.enemy_pos_array]=number_next*prob*prob
                                        new_board[x_next][y_next][self.player_pos_array] = nbr_monsters_enemy_cp[e]*(1-prob)*(1-prob)
                                        new_board[x][y][self.enemy_pos_array]-=number_next

                                #Updating monster arrays based on board updates
                                if flag_fight_enemy == 1:
                                    future_monsters_enemy = (1-prob_attacked)*(1-prob_attacked)*nbr_monsters_enemy_cp[e_attacked]
                                    if future_monsters_enemy == 0:
                                        pos_monsters_enemy_cp.pop(e_attacked)
                                        nbr_monsters_enemy_cp.pop(e_attacked)
                                    else:
                                        nbr_monsters_enemy_cp[e_attacked] = future_monsters_enemy

                                #If neither humans nor enemy monsters were in the next position then we just move our attacking monsters entirely
                                if flag_fight_humans == 0 and flag_fight_enemy == 0:
                                    new_board[x_next][y_next][self.enemy_pos_array]+=number_next
                                    new_board[x][y][self.enemy_pos_array]-=number_next

                    
                    #Call minimax alphabeta again with new updated board, with 1 depth in addition and changing to the other player (not is_max_turn)
                    eval_child, source_child, target_child = self.minimax_alphabeta(current_depth+1,new_board,not is_max_turn, alpha, beta)
                    
                    #We want to display the heuristic at the end of the analysis
                    if current_depth==0:
                        print("Evaluation for Monster1 {} for next move {}, and for Monster2 {} next move {}: {}"
                            .format(pos_monsters[0],next_moves_xy_total[0][m1],pos_monsters[1],next_moves_xy_total[1][m2],eval_child))

                    #Applying minimax alphabeta pruning
                    if is_max_turn and best_value < eval_child: #If our child evaluation is greater
                        best_value = eval_child #Update greatest evaluation
                        for monst in range(len(pos_monsters)):
                            if monst == 0:
                                idx = m1
                            elif monst == 1:
                                idx = m2                                
                            action_source[monst] = [pos_monsters[monst][0],pos_monsters[monst][1],nbr_monsters[monst]] #Save our best results for source
                            action_target[monst] = next_moves_xy_total[monst][idx] #Save our best results for target

                        alpha = max(alpha, best_value) #Update alpha
                        if beta <= alpha:
                            #Alpha-Beta Pruning algorithm
                            break_activated = 1
                            break

                    elif (not is_max_turn) and best_value > eval_child: #If our child evaluation is smaller
                        best_value = eval_child  #Update smallest evaluation
                        for monst in range(len(pos_monsters)):
                            if monst == 0:
                                idx = m1
                            elif monst == 1:
                                idx = m2                                
                            action_source[monst] = [pos_monsters[monst][0],pos_monsters[monst][1],nbr_monsters[monst]] #Save our best results for source
                            action_target[monst] = next_moves_xy_total[monst][idx] #Save our best results for target
                                                   
                        beta = min(beta, best_value) #Update beta
                        if beta <= alpha:
                            #Alpha-Beta Pruning algorithm
                            break_activated = 1
                            break

                #Alpha-Beta Pruning algorithm            
                if break_activated == 1:
                    break
        return best_value, action_source, action_target



    def play_ai(self):
        """
        Function that moves our monsters based on the heuristics of the board at each turn
        """  

        print("Game ON!")
        self.update_board() #Update Board initially
        self.play_number = 0 #Initialize number of play variable for a player

        while 1:
            self.play_number += 1 #Updates number of play
            print("TURN NUMBER",self.play_number)

            pos_humans = []
            nbr_humans = []
            coor_monsters = []
            nbr_monsters = []
            coor_monsters_enemy = []
            nbr_monsters_enemy = []   

            #Getting coordinates and number of monsters, enemies and humans in the board    
            for i in range(self.height):
                for j in range(self.width):
                    if self.board_challenge[i][j][0]>0:
                        pos_humans.append((i,j))
                        nbr_humans.append(self.board_challenge[i][j][0])       
                    elif self.board_challenge[i][j][self.player_pos_array]>0:
                        coor_monsters.append((i,j))
                        nbr_monsters.append(self.board_challenge[i][j][self.player_pos_array])
                    elif self.board_challenge[i][j][self.enemy_pos_array]>0:
                        coor_monsters_enemy.append((i,j))
                        nbr_monsters_enemy.append(self.board_challenge[i][j][self.enemy_pos_array])

            
            #Checking if there are still humans in the map or if there are many subgroups of monsters of the same kind
            if not (len(pos_humans) == 0 and len(coor_monsters) == 1):

                print("coor_monsters = {}".format(coor_monsters))
                print("nbr_monsters = {}".format(nbr_monsters))            
                print("coor_monsters_enemy = {}".format(coor_monsters_enemy))
                print("nbr_monsters_enemy = {}".format(nbr_monsters_enemy))  
                print("pos_humans = {}".format(pos_humans))
                print("nbr_humans = {}".format(nbr_humans))  
                        
                #Converting Board from list of lists of tuples TO list of lists of lists
                board_move = []
                for r in range(5):
                    board_challenge_row = [list(elem) for elem in self.board_challenge[r]]
                    board_move.append(board_challenge_row)

                eval_score, action_source, action_target = self.minimax_alphabeta(0,board_move,True,float('-inf'),float('inf'))

                print("action_source=",action_source)
                print("action_target=",action_target)

                #Move AI: the server has the x-y coordinate system switched
                self.move_ai(action_source, action_target) 
                #time.sleep(0.5) #Sleep a bit to see changes on the board                
                self.update_board() #Update Board
                #self.print_board() #Print Board

            #When no more humans and all monsters grouped together, we simply attack the closest enemy
            else:

                #Computes closest enemy
                dist_closest_enemy = 100
                for i in range(len(coor_monsters_enemy)):
                    distance_to_enemy = self.calculateDistance_plays(coor_monsters[0][0],coor_monsters[0][1],coor_monsters_enemy[i][0],coor_monsters_enemy[i][1])            
                    if distance_to_enemy < dist_closest_enemy:     
                        dist_closest_enemy = distance_to_enemy
                        x_enemy_min = coor_monsters_enemy[i][0]
                        y_enemy_min = coor_monsters_enemy[i][1]

                next_poss_moves = self.possible_nxt_moves(coor_monsters[0][0],coor_monsters[0][1])
                #Splitting Move List into tuples of 2 elements
                next_moves_xy = list(next_poss_moves[x:x + 2]  for x in range(0, len(next_poss_moves), 2))  
                random.shuffle(next_moves_xy) #Shuffle list of possible next moves

                #Computes best next move in order to get closer to enemy
                distance_to_enemy_min = 100
                for k in range(len(next_moves_xy)): 
                    distance_to_enemy = self.calculateDistance_plays(next_moves_xy[k][0],next_moves_xy[k][1],x_enemy_min,y_enemy_min)            
                    if distance_to_enemy < distance_to_enemy_min:     
                        distance_to_enemy_min = distance_to_enemy
                        x_next_to_enemy = next_moves_xy[k][0]
                        y_next_to_enemy = next_moves_xy[k][1]

                #Move AI: the server has the x-y coordinate system switched
                self.move_ai_single(coor_monsters[0][1],coor_monsters[0][0],nbr_monsters[0],y_next_to_enemy,x_next_to_enemy) 
                #time.sleep(1) #Sleep a bit to see changes on the board                
                self.update_board() #Update Board


def main():
    """
    Main function that calls the rest of the program and runs the game
    """      
    AI_Monster = Game()                
    AI_Monster.play_ai()

if __name__ == "__main__":
    main()