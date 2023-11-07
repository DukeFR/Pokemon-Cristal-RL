import cv2
import sys
import numpy
import numpy as np
import torch
import uuid
from pathlib import Path

from gymnasium import Env, spaces

from gymnasium.spaces import Box, Discrete
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv

from stable_baselines3.common.callbacks import CheckpointCallback

from skimage.transform import downscale_local_mean

from pyboy import PyBoy, WindowEvent



class CrystalEnv(Env):

    def __init__(self, headless = True, gb_path = './Crystal.gbc', debug = False):
        quiet = "--quiet" in sys.argv

        self.pyboy = PyBoy(gb_path)
        self.pyboy.map_emulation_speed(5)

        file_like_object = open("go.state", "rb") #load save state

        self.pyboy.load_state(file_like_object)
        
        self.curFight = 0 #0 if not in Fight / 1 in Fight
        self.level1 = 0 #Level of the first Pokemon
        self.map = np.zeros((600,600,600), dtype = np.uint8) #Array of [idmap][x][y]
        self.max_steps = 20480 #Max steps before done
        self.action_space = Discrete(6) #All actions (up, down, left, right, a, b)


        self.observation_space = spaces.Dict(
            {
                "Image": Box(low=0, high=255, shape=(72, 80, 3), dtype=np.uint8), #Screen of GB
                #"map": Box(low=0,high=1, shape = (50,100,100), dtype=np.uint8), #Coordinates of map
                "fight": spaces.Box(low=0, high=1, dtype = np.uint8), #If in fight or not
                "up": spaces.Box(low=0, high=1, dtype = np.uint8), #If there is collision if going up
                "down": spaces.Box(low=0, high=1, dtype = np.uint8), #Same but for down
                "right": spaces.Box(low=0, high=1, dtype = np.uint8), #Same but for right
                "left": spaces.Box(low=0, high=1, dtype = np.uint8) #Same but for left

            }

        )

        self.reward = 0 #Reward

        self.output_shape = (72, 80, 3) #Shape of the image of GB

        self.wild = 0 #0 if not wild fight / 1 if in wild fight
        self.hp = -100 #Get hp of the ennemy

        self.left = 0 #collision left
        self.right = 0 #collision right
        self.up = 0 #collision up
        self.down = 0 #collision down

        self.xp1 = 0 #get xp value of first pokemon
        self.step_count = 0 #current step count

        self.percentage = 0 #get percentage of health bar of the ennemy


    def reset(self, seed=None):
        file_like_object = open("go.state", "rb") #reset to state
        self.map = np.zeros((600,600,600), dtype=np.uint8)
        self.reward = 0
        self.wild = 0
        self.step_count = 0

        self.left = 0
        self.right = 0
        self.up = 0
        self.down = 0

        self.curFight = 0
        self.percentage = 0
        self.hp = -100
        self.xp1 = 0

        self.level1 = 0
        dict = {}
        self.pyboy.load_state(file_like_object)

        game_pixels_render = numpy.array(self.pyboy.screen_image())
        obs = (downscale_local_mean(game_pixels_render, (2,2,1))).astype(np.uint8)
        observation_space = {
                "Image": obs,
                #"map": self.map,
                "fight": np.array([self.curFight], dtype = np.uint8),
                "up": np.array([self.up], dtype = np.uint8),
                "down": np.array([self.down], dtype = np.uint8),
                "left": np.array([self.left], dtype = np.uint8),
                "right": np.array([self.right], dtype = np.uint8)
            }

        return observation_space, dict



    def step(self, action):

        OldCoordinates = (self.pyboy.get_memory_value(56504), self.pyboy.get_memory_value(56503)) #Coordinates of the player before moving
        self.level1 = self.pyboy.get_memory_value(56574) #Level of 1st pokemon before moving


        self.up = 0 #Reset collision
        self.down = 0
        self.left = 0
        self.right = 0

        if action == 0:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_A)
            self.pyboy.tick()
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_A)
            self.pyboy.tick()

        if action == 1:
            self.pyboy.send_input(WindowEvent.PRESS_BUTTON_B)
            self.pyboy.tick()
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_B)
            self.pyboy.tick()

        if action == 2:
            collision = self.pyboy.get_memory_value(49915)
            if(collision!=0): #If collision up
                if(collision!=113): #If not door or stairs
                    if(collision!=122):
                        self.up = 1
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_UP)
            self.pyboy.tick()
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_UP)
            self.pyboy.tick()


        if action == 4:
            collision = self.pyboy.get_memory_value(49914)
            if(collision!=0):
                if(collision!=255):
                    self.down = 1
            self.pyboy.send_input(WindowEvent.PRESS_ARROW_DOWN)
            self.pyboy.tick()
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_DOWN)
            self.pyboy.tick()


        if action == 3:
            collision = self.pyboy.get_memory_value(49916)
            if(collision!=0):
                self.left = 1

            self.pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
            self.pyboy.tick()
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)
            self.pyboy.tick()



        if action == 5:
            collision = self.pyboy.get_memory_value(49917)
            if(collision!=0):
                self.right = 1

            self.pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
            self.pyboy.tick()
            self.pyboy.tick()
            self.pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)
            self.pyboy.tick()

        newlvl = self.pyboy.get_memory_value(56574) #Check new level after action made

        self.wild = self.pyboy.get_memory_value(53805) #Check if in wild fight
        coordinates = ((self.pyboy.get_memory_value(56504), self.pyboy.get_memory_value(56503))) #New Coordinates
        pos = self.pyboy.get_memory_value(53576) #Get map level

        if int(self.pyboy.get_memory_value(53805) == 1): #If in wild fight
                if(int(self.pyboy.get_memory_value(53783)!=0)):

                    if(self.curFight == 0): #If step before was not in fight
                        print("COMBAT")
                        self.curFight = 1 #fight begins!
                        self.hp = int(self.pyboy.get_memory_value(53783)) #get hp of ennemy
                        self.up = 0 #reset collision to simplify the analysis
                        self.down = 0
                        self.left = 0
                        self.right = 0

                    if(self.hp-int(self.pyboy.get_memory_value(53783)) !=0): #if the hp at the previous step is different to now (for the ennemy)
                        temp = round(float(int(self.pyboy.get_memory_value(53783)) / int(self.hp)),1) #Get current hp of ennemy
                        if(temp!=self.percentage and abs(self.percentage - temp) > 0.1 and temp < 1): #If attack has done enough damage
                            print("ATTACK")
                            self.percentage = temp
                            print("avc: " + str(self.percentage)) #Get current percentage of the health bar of the ennemy
                            self.reward += abs(1-temp) * 0.002 #Get reward depending of how efficient the attack was


        if(self.curFight==1): #If in fight
            if(newlvl == self.level1+1): #If level is different than previous step
                print("LEVEL UP")
                print("new:" + str(newlvl))
                print("old:" + str(self.level1)) #Level UP!
                self.reward += newlvl * 0.004 #Reward increased depending on how high level is the poke 1

        if(int(self.pyboy.get_memory_value(56555)) != self.xp1) and int(self.pyboy.get_memory_value(56555)) !=0 and int(self.pyboy.get_memory_value(56555)) !=127: #If we kill the ennemy (we get xp)
            if self.curFight == 1 and self.xp1 < int(self.pyboy.get_memory_value(56555)):
                print("KILL")
                print(str(self.pyboy.get_memory_value(56555)))
                self.xp1 = int(self.pyboy.get_memory_value(56555))
                self.reward += 0.003 #Reward for killing


        if(coordinates != OldCoordinates): #If we have move of coordinates then we're not in fight anymore
            if (coordinates != (0,0) and OldCoordinates != (0,0)): #Reset fight stats
                self.curFight = 0
                self.percentage = 0
                self.hp = -100

        game_pixels_render = numpy.array(self.pyboy.screen_image())


        obs = (downscale_local_mean(game_pixels_render, (2,2,1))).astype(np.uint8) #Rendering gb for obs

        observation_space = {
                "Image": obs,
                #"map": self.map,
                "fight": np.array([self.curFight], dtype = np.uint8),
                "up": np.array([self.up], dtype = np.uint8),
                "down": np.array([self.down], dtype = np.uint8),
                "left": np.array([self.left], dtype = np.uint8),
                "right": np.array([self.right], dtype = np.uint8),

            }

        info = {} #Don't know what to do with that but hey!

        booleen = 0 #Bool to check if we have already been there!

        if(self.map[pos][coordinates[0]][coordinates[1]] != 0.0): #If we've been there
            booleen = 1

        if(booleen==0):
            self.map[pos][coordinates[0]][coordinates[1]] = 1
            self.reward = self.reward + 0.008 #If not then we add a 1 and add reward

        done = self.step_count >= self.max_steps - 1 #Done if step max reached
        self.step_count += 1 #add step

        return observation_space,self.reward,False,done,info #Return everything yay!




def predict(): #Mostly used for testing purpose and playing in the emu
    env = CrystalEnv()
    #model = PPO("MlpPolicy", env, verbose=1)
    #model = PPO.load("TEMP0.zip", env)
    hp = -100
    curFight = 0
    Oldcoordinates = []
    coordinates = []
    file_like_object = open("adventure.state", "rb")
    env.pyboy.load_state(file_like_object)
    while not env.pyboy.tick() :

        #print(str(int(env.pyboy.get_memory_value(53805))))
        Oldcoordinates = ((env.pyboy.get_memory_value(56504), env.pyboy.get_memory_value(56503)))
        #print(str(env.pyboy.get_memory_value(56504)) + " + " + str(env.pyboy.get_memory_value(56503)))
        raw = numpy.array(env.pyboy.screen_image())[:,:,:]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        obs = gray.astype(np.uint8)
        #env.pyboy.save_state(file_like_object)
            #print("boucle")
        #action = model.predict(obs, deterministic=False)
        #print(action[0])
        #for act, prob in enumerate(action):
            #print(f"Action {act}: " + str(prob))
        #obs,reward,done,truncated,info = env.step(action[0])
        #print("---")
        #print(reward)


def evaluate(): #Not really using it but not clearing it can be useful later on
    env = CrystalEnv()
    #model = PPO.load("",env=env)
    #mean_reward, std_reward = evaluate_policy(model,env,n_eval_episodes=2)
    #print("mean r:" + str(mean_reward))
    #print("r: " + str(std_reward))

def launcher(): #Main algorithm !!!
    ep_length = 4096*10
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')

    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,name_prefix='poke')

    envs = make_vec_env(env_id=CrystalEnv, seed=1, n_envs= 5, vec_env_cls=SubprocVecEnv) #Change n_envs for move env!
    callback = EvalCallback(envs, eval_freq=5000, n_eval_episodes=2, best_model_save_path="./models/", verbose=1) #Eval callback not used :(

    #WARNING : USING CUDA! If not change to device CPU


    #MultiInputPolicy (currently trying that!)
    model = PPO("MultiInputPolicy", envs,learning_rate=0.0001, verbose=1, ent_coef=0.2, n_steps=ep_length, batch_size=512, n_epochs=1, gamma=0.997,  tensorboard_log=sess_path, device="cuda")

    #load prev model
    #model = PPO.load("", env = envs)

    #CNN Policy (i've also tried MLPPolicy)
    #model = PPO(policy= "CnnPolicy", ent_coef=0.1, n_steps = 16384, clip_range=0.3, n_epochs=3, batch_size=512, gamma=0.999,env=envs, device="cuda")

    for i in range(40): #Learning
        model.learn(total_timesteps=204800, log_interval=1, tb_log_name=str(0), callback=checkpoint_callback,remap_num_timesteps= False)
        print(envs.get_attr("reward"))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #MAIIIN check if cuda is available

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    #env_checker.check_env(CrystalEnv())
    launcher()

