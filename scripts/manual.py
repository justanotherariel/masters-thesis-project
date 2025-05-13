from minigrid.manual_control import ManualControl
import minigrid
import gymnasium as gym
import pygame

env_name = "MiniGrid-KeyCorridorS6R3-v0"

# Initialize Pygame
pygame.init()

# Create the environment
env = gym.make(env_name, render_mode="human")

# Start manual control
manual_control = ManualControl(env.unwrapped)
manual_control.start()