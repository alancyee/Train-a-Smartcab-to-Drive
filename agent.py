import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.build_state_to_index_dict()
        self.q_table = [[0 for k in range(4)] for i in range(192)]        
        self.epsilon = 900
        self.alpha = .95
        self.gamma = .05
        self.reached_destination = 0
        
    
    def build_state_to_index_dict(self):
        self.d = {x: {y: {z: {w: {v: 0 for v in ['normal','rush']} for w in [None, 'forward', 'left', 'right']} for z in [None, 'forward', 'left', 'right']} for y in ['red', 'green']} for x in ['forward', 'left', 'right']}
        count = 0
        for key1 in self.d.keys():
            for key2 in self.d[key1].keys():
                for key3 in self.d[key1][key2].keys():
                    for key4 in self.d[key1][key2][key3].keys():
                        for key5 in self.d[key1][key2][key3][key4].keys():
                            self.d[key1][key2][key3][key4][key5] = count
                            count += 1
                            
    def get_state_index(self):
        if self.deadline <= 5:
            deadline = 'rush'
        else:
            deadline = 'normal'
        self.state = self.d[self.next_waypoint][self.light][self.oncoming][self.left][deadline]
        return self.state
        
    def get_action_index(self, action):
        self.action_to_index_dict = {None: 0, 'forward': 1, 'right': 2, 'left': 3}
        return self.action_to_index_dict[action]
        
    def get_q_value(self, action, state=None):     
        if not state:
            return self.q_table[self.get_state_index()][self.get_action_index(action)]
        else:
            return self.q_table[self.get_next_state_index(state)][self.get_action_index(action)]
    
    def get_action(self):
        if random.randint(1,1000) <= self.epsilon:
            self.epsilon -= 1
            return random.choice([None, 'forward', 'left', 'right'])
        else:        
            return self.get_highest_q_value_action() 
    
    def get_highest_q_value_action(self):
        highest_value = -10
        highest_q_value_actions = []
        for action in [None, 'forward', 'left', 'right']:
            if self.get_q_value(action) > highest_value:
                highest_value = self.get_q_value(action)
                highest_q_value_actions = [action]
            elif self.get_q_value(action) == highest_value:
                highest_q_value_actions.append(action)
        return random.choice(highest_q_value_actions)
            
                               
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
    
    def get_next_state_q_max(self, action):
        chance_red_light = .5
        chance_green_light = .5
        if action == None:
            next_waypoint = self.next_waypoint
            if self.light == 'green':
                chance_green_light = self.get_odds_light_remains_same_color()
                chance_red_light = 1 - chance_green_light
            else:
                chance_red_light = self.get_odds_light_remains_same_color()
                chance_green_light = 1 - chance_red_light
        #generealize next_waypoint is forward
        else:
            next_waypoint = 'forward'
        if self.deadline <= 6:
            deadline = 'rush'
        else:
            deadline = 'normal'    
        q_max = -10        
        for action in [None, 'forward', 'left', 'right']:
            state = [next_waypoint, 'green', None, None, deadline]            
            green_score = self.get_q_value(action, state)  * chance_green_light
            state = [next_waypoint, 'red', None, None, deadline]            
            red_score = self.get_q_value(action, state)  * chance_red_light
            weighted_score = red_score + green_score
            if (weighted_score) > q_max:
                q_max = weighted_score
        return q_max            
        
    def get_odds_light_remains_same_color(self):
        #light has 1/3 chance of having a light cycle of 3,4,5 time units
        #if a light has a 3 time unit cycle, you have a 2/3 chance it stays same
        return 2 * 1.0 / 3 / 3  + .75 / 3 + .8 / 3                      
        
    def get_next_state_index(self, state):
        return self.d[state[0]][state[1]][state[2]][state[3]][state[4]]       
    
    def update_q_table(self, action, reward):
        q_old = self.get_q_value(action)
        q_new = (1 - self.alpha) * q_old + self.alpha * (reward + self.gamma * self.get_next_state_q_max(action)) 
        self.q_table[self.get_state_index()][self.get_action_index(action)] = q_new

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.deadline = deadline
        self.light = inputs['light']        
        self.oncoming = inputs['oncoming']        
        self.left = inputs['left']
                
        # TODO: Select action according to your policy
        action = self.get_action()

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward > 8:
            self.reached_destination += 1

        # TODO: Learn policy based on state, action, reward
        self.update_q_table(action, reward)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1)  # reduce update_delay to speed up simulation
    #tune_parameters(a, sim)
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

def tune_parameters(agent, sim):
    a = agent    
    max_reached = 0    
    best_parameters = []
    
    for gamma in range (0,21,1):
        gamma = gamma / 20.0
        for alpha in range (0,21,1):
            alpha = alpha / 20.0
#    for gamma in [.3,.7]:        
#        for alpha in [.3,.7]:            
            for epsilon in [900,1000,1100,1300,1400,1500]:
                a.reached_destination = 0
                a.epsilon = epsilon
                a.alpha = alpha
                a.gamma = gamma                
                sim.run(n_trials=100)
                if a.reached_destination > max_reached:
                    max_reached = a.reached_destination
                    best_parameters = [(epsilon, alpha, gamma, max_reached)]
                elif a.reached_destination == max_reached:
                    max_reached = a.reached_destination
                    best_parameters.append((epsilon, alpha, gamma, max_reached))   
            print '\n'            
            print best_parameters
            print '\n'

if __name__ == '__main__':
    run()
