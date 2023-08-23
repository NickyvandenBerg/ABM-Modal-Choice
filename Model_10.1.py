# $ conda activate spyder-env

#%% Importing libraries
import mesa
import mesa.time as time
import mesa.space as space
from mesa.datacollection import DataCollector
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import math
import numpy as np
import pandas as pd
import pickle
import networkx as nx
import random
import geopy.distance
import plotly
import seaborn as sns 
from mpl_toolkits.basemap import Basemap as Basemap
import matplotlib.pyplot as plt
from scipy import stats as stats

from scipy.stats import beta as beta

# data
df = pd.read_csv("train_stations_europe_capitals2.csv", delimiter=';')


latitude = df.loc[:, 'latitude']
longitude = df.loc[:,'longitude']


coord_dic = {}
for index, row in df.iterrows():
    label = row['city']
    lon = row['longitude']
    lat = row['latitude']
    coord_dic[label] = (lon, lat)

pos = {label: (x, y) for label, (x, y) in coord_dic.items()} #does same as above


# Network
from math import radians, sin, cos, sqrt, atan2

def calculate_distance(lat1, lon1, lat2, lon2):
    # Earth's radius in kilometers
    radius = 6371

    # Convert latitude and longitude from degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Difference between the latitudes and longitudes
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Calculate the distance
    distance = radius * c

    return distance


ax = plt.figure(figsize=(13, 13))

basemap = Basemap(
    projection = 'merc',
    llcrnrlon = -11, 
    urcrnrlon = 25,     
    llcrnrlat = 35,  
    urcrnrlat = 65,    
    lat_ts = 0,
    resolution = 'l',
    suppress_ticks = True)

basemap_x, basemap_y = basemap(df['longitude'].values, df['latitude'].values)

# graph
G = nx.Graph()
for node, coords in coord_dic.items():
    lon, lat = coords
    G.add_node(node, pos=(lon, lat)) #(x, y)

# (x, y or latitude, longitude) is consistent with the distance calculation
threshold = 800 
for node1, coord1 in coord_dic.items():
    for node2, coord2 in coord_dic.items():
        if node1 != node2:
            lon1, lat1 = G.nodes[node1]['pos']
            lon2, lat2 = G.nodes[node2]['pos']
            distance = calculate_distance(lat1, lon1, lat2, lon2)
            #print(distance)
            if distance < threshold:  
                G.add_edge(node1, node2)

G.remove_edge('London', 'Amsterdam')
G.remove_edge('London', 'Zurich')
G.remove_edge('Copenhagen', 'Warsaw')
G.remove_edge('Berlin', 'Vienna')
G.remove_edge('Amsterdam', 'Paris')
G.add_edge("Madrid", "Paris")
G.add_edge("Zurich", "Madrid")

pos = {}
for i, city in enumerate (df['city']):
    pos[city] = (basemap_x[i], basemap_y[i])

nx.draw(G, pos, with_labels=True, font_size = 20, node_size=15, node_color = 'r', edge_color='g', width=0.6)

basemap.drawcoastlines(linewidth = 0.5)
plt.tight_layout()

plt.show()


with open('rail_network.data', 'wb') as file:
    pickle.dump(G, file)

#%%MODEL

# AGENT CLASS
class TravelAgent (mesa.Agent):
    def __init__(self, unique_id, model, origin_position, age, income, env_aware): 
        super().__init__(unique_id, model)
        self.destination = None
        self.origin_position = origin_position
        self.age = age
        self.modal_choice = None
        self.income = income
        self.env_aware = env_aware
        self.trainpath = []
        self.airpath = []
        self.path_step = 0
        self.trainscore = 0
        self.airplanescore = 0
        self.modalchoice = self.function_modal_choice()
        self.train_distance = 0
        self.air_distance = 0
        self.train_time = 0
        self.air_time = 0
        self.train_price = 0
        self.air_price = 0
        self.air_price_old = 0
        self.air_price_new = 0
        self.train_emissions = 0
        self.air_emissions = 0
        self.train_convenience = 0
        self.air_convenience = 0
        self.travelled_distance = 0
        self.carbontax_trainprice = 0
        self.carbontax_airprice = 0
        self.pos = None
        self.carbon_emission_tax = False
        self.carbon_fuel_tax = False
        self.carbon_flatticket_tax = False
        self.carbon_distanceticket_tax = True
        
    
    def assign_destination(self, destination):
        if destination != self.origin_position and destination != self.pos:  # Check if destination is different from the current position
            self.destination = destination    
        
    def path_train(self):
        if self.model.schedule.steps == 0: 
            if self.destination is not None: #if not self.path: #checks if the self path list is empty
                self.trainpath = nx.shortest_path(self.model.network, source=self.origin_position, target=self.destination, method='dijkstra')
                self.path_step += 1 #not necessary?
                    
    def move_train(self):
        if self.model.schedule.steps > 0:
            if len(self.trainpath) > 1 and self.path_step < len(self.trainpath):
                next_node = self.trainpath[self.path_step]
                self.model.grid.move_agent(self, next_node)
                self.pos = next_node
                self.path_step += 1 

    def path_airplane(self):
        if self.model.schedule.steps == 0:
            if self.destination is not None:
                self.airpath = (self.origin_position, self.destination)
                self.path_step += 1 #not necessary?
    
    def move_airplane(self):
        if self.model.schedule.steps > 0:
            self.model.grid.move_agent(self, self.destination)
            self.pos = self.destination

    def distance_calculator (self, lat1, lon1, lat2, lon2):
        # Earth's radius in kilometers
        radius = 6371.0
    
        # Convert latitude and longitude from degrees to radians
        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)
    
        # Difference between the latitudes and longitudes
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
    
        # Haversine formula
        a = sin(dlat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
        # Calculate the distance
        distance = radius * c
    
        return distance
    
        
    def calculate_train_distance(self):
        if self.model.schedule.steps == 0:
            pos = nx.get_node_attributes(self.model.network, "pos")
            
            for i in range(len(self.trainpath)-1):
                stad_0 = self.trainpath[i]
                lon1, lat1 = pos[stad_0] 
                stad_1 = self.trainpath[i+1]
                lon2, lat2 = pos[stad_1]
    
                self.train_distance += self.distance_calculator(lat1, lon1, lat2, lon2)

            return self.train_distance
        
    def calculate_air_distance(self):
        if self.model.schedule.steps == 0:
            pos = nx.get_node_attributes(self.model.network, "pos")
            
            for i in range(len(self.airpath)-1):
                stad_0 = self.airpath[i]
                lon1, lat1 = pos[stad_0] 
                stad_1 = self.airpath[i+1]
                lon2, lat2 = pos[stad_1] 
                
                self.air_distance = self.distance_calculator(lat1, lon1, lat2, lon2)
                
            return self.air_distance

    def pert_random(self, minimum, mode, maximum):
        # approximation by Beta distribution
        lmbd = 4 #if you want to adjsut this per variable, put it in the def
        
        alpha_param = 1 + lmbd * ((mode - minimum) / (maximum - minimum))
        beta_param = 1 + lmbd * ((maximum - mode) / (maximum - minimum))

        return beta.rvs(alpha_param, beta_param, loc=minimum, scale=maximum-minimum)
    
    def train_time_calculator(self):
        if self.model.schedule.steps == 0:
            #trainspeed = 250 #km/h
            trainspeed = self.pert_random(100, 120, 320)
            self.train_time = self.train_distance / trainspeed
            return self.train_time
    
    def air_time_calculator(self):
        if self.model.schedule.steps == 0:
            #airspeed = 800 #km/h
            airspeed = self.pert_random(740, 780, 930)
            self.air_time = self.air_distance / airspeed
            return self.air_time
  
    def calculate_train_emissions(self):
        if self.model.schedule.steps == 0:
            emissionkm = 0.019 #CO2e / passenger km
            self.train_emissions = self.train_distance * emissionkm
            return self.train_emissions
    
    def calculate_air_emissions(self):
        if self.model.schedule.steps == 0:
            emissionkm = 0.123 #CO2e / passenger km
            self.air_emissions = self.air_distance * emissionkm
            return self.air_emissions
    
    def air_price_calculator(self):
        if self.model.schedule.steps == 0:
            if self.air_distance < 1000:
                pricekm =self.pert_random(0.05, 0.12, 0.19)
            else:
                pricekm =self.pert_random(0.05, 0.06, 0.19)
            
            self.air_price = self.air_distance * pricekm
            self.air_price_old = self.air_price
            
            self.carbon_tax() 
            
            return self.air_price
    
    def train_price_calculator(self):
        if self.model.schedule.steps == 0:
            #pricekm = 0.20
            if self.train_distance < 1000:
                pricekm = self.pert_random(0.10, 0.15, 0.30)
                self.train_price = self.train_distance * pricekm
            elif self.train_distance >= 1000:
                pricekm = self.pert_random(0.10, 0.12, 0.30)
                self.train_price = self.train_distance * pricekm
            return self.train_price
    
    def train_convenience_calculator(self):
        if self.model.schedule.steps == 0:
            pert_val = self.pert_random(0.1, 0.78, 1) #random choice from pert distribution
            self.train_convenience = min(max(pert_val, 0.1), 1) #limited to never go below 0.1 or above 0.9
            return self.train_convenience
    
    def air_convenience_calculator(self):
        if self.model.schedule.steps == 0:
            pert_val = self.pert_random(0.1, 0.22, 1)
            self.air_convenience = min(max(pert_val, 0.1), 1) 
            return self.air_convenience

    def carbon_tax(self):
        if self.model.schedule.steps != 0:
            return
        t_carbon_emission = 0.04466 #€44.66 / 1000 kg CO2
        
        "change fuel tax by blocking out the other values"
        t_carbon_fuel = 0.067       #€67 / 1000 kg CO2
        #t_carbon_fuel = 0.131      #€131 / 1000 kg CO2
        #t_carbon_fuel = 0.198      #€198 / 1000 kg CO2
        
        t_carbon_flatticket = 10.43
        
        t_carbon_ticket_short = 25.30
        t_carbon_ticket_long = 10.12
        
        if self.carbon_emission_tax == True:
            extra_train_price = self.train_emissions * t_carbon_emission
            self.train_price += extra_train_price
            self.carbontax_trainprice = extra_train_price
    
            extra_air_price = self.air_emissions * t_carbon_emission
            self.air_price += extra_air_price
            self.carbontax_airprice = extra_air_price

            
        elif self.carbon_fuel_tax == True:
            carbontax_airprice = self.air_emissions * t_carbon_fuel
            self.carbontax_airprice = carbontax_airprice
            self.air_price += self.carbontax_airprice
            self.air_price_new = self.air_price
            carbontax_airprice = self.air_emissions * t_carbon_fuel
            
        elif self.carbon_flatticket_tax == True:
            self.air_price += t_carbon_flatticket
            self.carbontax_airprice = t_carbon_flatticket
        
        elif self.carbon_distanceticket_tax == True:
            if self.air_distance <= 350: 
                self.air_price += t_carbon_ticket_short
                self.carbontax_airprice = t_carbon_ticket_short
            if self.air_distance > 350: 
                self.air_price += t_carbon_ticket_long
                self.carbontax_airprice = t_carbon_ticket_long
        
    def calculate_train_score(self): #weight * relative value * characteristic value
        if self.model.schedule.steps == 0:
            #relative values
            price_train = self.train_price_calculator()
            price_air = self.air_price_calculator()
            rel_price = price_train / (price_train + price_air)
            
            
            emissions_train = self.calculate_train_emissions()
            emissions_air = self.calculate_air_emissions()
            rel_emissions = emissions_train / (emissions_train + emissions_air)
            
            time_train = self.train_time_calculator()
            time_air = self.air_time_calculator()
            rel_time = time_train / (time_train + time_air)
            
            convenience_train = self.train_convenience_calculator()
            convenience_air = self.air_convenience_calculator()
            rel_convenience = convenience_train / (convenience_train + convenience_air)
            
            #Characteristic values
            if self.age == 'Young Adult':
                val_age = 0.2
            if self.age =='Adult':
                val_age = 0.5
            if self.age =='Senior':
                val_age = 0.8
            
            if self.income == 'Low':
                val_income = 0.8
            if self.income == 'Medium':
                val_income = 0.5
            if self.income == 'High':
                val_income = 0.2
                
            if self.env_aware == 'Low':
                val_env_aware = 0.2
            if self.env_aware =='Medium':
                val_env_aware = 0.5
            if self.env_aware == 'High':
                val_env_aware = 0.8
                    
            #trainscore equation
            trainscore = rel_price * val_income + rel_time * (1 - val_age) + rel_emissions * val_env_aware + rel_convenience * val_age
            self.trainscore += trainscore
            
            return self.trainscore

    def calculate_airplane_score(self):
        if self.model.schedule.steps == 0:
            #relative values
            price_train = self.train_price_calculator()
            price_air = self.air_price_calculator()
            rel_price = price_air / (price_air + price_train)
            
            
            emissions_train = self.calculate_train_emissions()
            emissions_air = self.calculate_air_emissions()
            rel_emissions = emissions_air / (emissions_train + emissions_air)
            
            time_train = self.train_time_calculator()
            time_air = self.air_time_calculator()
            rel_time = time_air / (time_train + time_air)
            
            convenience_train = self.train_convenience_calculator()
            convenience_air = self.air_convenience_calculator()
            rel_convenience = convenience_air / (convenience_train + convenience_air)
            
            #Characteristic values
            if self.age == 'Young Adult':
                val_age = 0.2
            if self.age =='Adult':
                val_age = 0.5
            if self.age =='Senior':
                val_age = 0.8
            
            if self.income == 'Low':
                val_income = 0.8
            if self.income == 'Medium':
                val_income = 0.5
            if self.income == 'High':
                val_income = 0.2
                
            if self.env_aware == 'Low':
                val_env_aware = 0.2
            if self.env_aware =='Medium':
                val_env_aware = 0.5
            if self.env_aware == 'High':
                val_env_aware = 0.8
            
            #airscore equation          
            airplanescore = rel_price * val_income + rel_time * (1 - val_age) + rel_emissions * val_env_aware + rel_convenience * val_age
            self.airplanescore += airplanescore

            return self.airplanescore
    
    def function_modal_choice(self):
        
        if self.airplanescore < self.trainscore:
            modal_choice = 'airplane'
            self.modalchoice = modal_choice
            return self.modal_choice
          
        elif self.airplanescore > self.trainscore:
            modal_choice = 'train'
            self.modalchoice = modal_choice
            return self.modal_choice

        else:
            modal_choice = 'Equal'
            self.modalchoice = modal_choice
            return self.modal_choice
        
    
    def step(self):
        if self.destination is None: # Assign a random (new) destination not equal to current position
            all_nodes = list(self.model.network.nodes())
            available_nodes = all_nodes
            available_nodes.remove(self.origin_position) #remove own location from copy of all nodes to avoid data loss
            self.assign_destination(random.choice(available_nodes))
        
        self.path_train()
        self.path_airplane()
        self.calculate_train_distance()
        self.calculate_air_distance()
        self.train_convenience_calculator()
        self.air_convenience_calculator()
        self.train_price_calculator()
        self.air_price_calculator()
        self.calculate_train_emissions()
        self.calculate_air_emissions()
        self.carbon_tax()
        self.calculate_train_score()
        self.calculate_airplane_score()
        self.function_modal_choice()

        if self.modalchoice == 'train':
            self.move_train()
            
        elif self.modalchoice == 'airplane':
            self.move_airplane()
            
        else: #what happens if scores are equal
            print("No valid mode of transportation chosen.")
            
        

    def path_agents(self):
        return None
    

# MODEL CLASS
class TravelModel(mesa.Model):
    def __init__(self):
        self.num_agents = df['agents'].sum()#every time we create a model we decide on the numb of agents
        self.agents = [] #empty list of agents
        self.agent_id_counter = 0
        self.origin_position = None
        self.step_count = 0

        # Grid        
        self.schedule = mesa.time.RandomActivation(self) 
        with open('rail_network.data', 'rb') as file: #rb?????
            self.network = pickle.load(file)   
       
        self.grid = space.NetworkGrid(self.network)
        
        # charactereistics
        age_list = ['Young Adult','Adult','Senior']
        income_list = ['Low','Medium','High']
        env_aware_list = ['Low','Medium','High']
        
        # Create agents
        for i, row in df.iterrows():
            location = row['city']  
            num_agents_at_location = row['agents']
        
            for i in range(num_agents_at_location):
                age = np.random.choice(age_list, p=[(0.191), (0.483), (0.326)]) 
                income = np.random.choice(income_list, p=[(0.165), (0.641), (0.194)])
                env_aware = np.random.choice(env_aware_list, p=[(0.06), (0.41), (0.53)])
                
                a = TravelAgent(self.agent_id_counter, self, location, age, income, env_aware)  
                self.agent_id_counter += 1
                a.function_modal_choice()  # modal_choice function, so it's determined later
                self.schedule.add(a)
                self.grid.place_agent(a, location)
                self.agents.append(a)
         
        #metrics
        #model_metrics = {"Modal choice of Agents": modal_choice}
        model_metrics = {"Number of Agents": count_agents}
                         #"Train travellers": train_travellers,
                         #"Air travellers": air_travellers}
        #agent_metrics = {"Path of Agents": path_agents, "Agent ID": "unique_id"}
        agent_metrics = {"Origin location": lambda agent: agent.origin_position,
                         "Destination": lambda agent: agent.destination,
                         "Age": lambda agent: agent.age,
                         "Income": lambda agent: agent.income,
                         "Environmentall Awareness": lambda agent: agent.env_aware,
                         "Train path of Agents": lambda agent: agent.trainpath,
                         "Air path of Agents": lambda agent: agent.airpath,
                         "Train price": lambda agent: agent.train_price,
                         "Air price": lambda agent: agent.air_price,
                         #"Old Air price": lambda agent: agent.air_price_old,
                         #"New Air price": lambda agent: agent.air_price_new,
                         "CT train price": lambda agent: agent.carbontax_trainprice,
                         "CT air price": lambda agent: agent.carbontax_airprice,
                         "Train emissions": lambda agent: agent.train_emissions,
                         "Air emissions": lambda agent: agent.air_emissions,
                         "Train time": lambda agent: agent.train_time,
                         "Air time": lambda agent: agent.air_time,
                         "Train convenience": lambda agent: agent.train_convenience,
                         "Air convenience": lambda agent: agent.air_convenience,
                         "Modal choice of Agents": lambda agent: agent.modalchoice,
                         "Trainscore": lambda agent: agent.trainscore,
                         "Airscore": lambda agent: agent.airplanescore,
                         "Travelled Distance air": lambda agent: agent.air_distance,
                         "Travelled Distance train": lambda agent: agent.train_distance}
        
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)
    
    def get_agent_characteristics(self):
        agent_characteristics = {}
        for agent in self.agents:
            agent_id = agent.unique_id
            age = str(agent.age)
            income = str(agent.income)
            env_aware = str(agent.env_aware)
            distance_travelled_air = agent.calculate_air_distance()
            distance_travelled_train = agent.calculate_train_distance()
            airpath_agent = str(agent.path_airplane())
            trainpath_agent = str(agent.path_train())
            
            agent_characteristics[agent_id] = {'age': age, 'income': income, "env aware": env_aware, "air distance travelled": distance_travelled_air, "train distance travelled": distance_travelled_train, "air path agent": airpath_agent, "train path agent": trainpath_agent}
        return agent_characteristics
    
    def step(self):
        print("This is step: " + str(self.schedule.steps))
        if self.step_count == 9:
            self.datacollector.collect(self) #collect in last step
        self.step_count += 1
        self.schedule.step()

#why do you need this?
def count_agents(self):
    return self.num_agents 


model = TravelModel() 
for i in range(10):
    model.step()

model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()


print(model_data)
print(agent_data)

agent_data.to_excel("output_data.xlsx")
#agent_data.to_csv("output_data", index = False)



