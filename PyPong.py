# Creator bearpaw7
# https://github.com/bearpaw7/PyPong
# Edited version for dueling AI.

'''
Created on Aug 30, 2012

@author: bearpaw7
'''
# Sample Python/Pygame Programs
# Simpson College Computer Science
# http://cs.simpson.edu

# Import a library of functions called 'pygame'
import os, math, pygame, random
import numpy as np

# Define the colors we will use in RGB format
black = (0, 0, 0)
white = (255, 255, 255)
blue =  (0, 0, 255)
red =   (255, 0, 0)

MOVEMENT_SPEED = 5

BALL_SAVING_REWARD = 5.0
BALL_DISTANCE_REWARD = 10.0
GOAL_REWARD = 10.0
GOAL_PUNISHMENT = 10.0

width = 640
height = 480

class Ball:
	base_velocity = 3.00
	def __init__(self,x,y,degrees):
		self.last_scorer = 0		# 1 Red, 2 Blue
		self.position=[x,y]
		self.direction=degrees
		self.radius=7
		self.velocity=self.base_velocity
		pygame.mixer.init()

	def top(self):
		return self.position[1]-self.radius
	def bottom(self):
		return self.position[1]+self.radius
	def left(self):
		return self.position[0]-self.radius
	def right(self):
		return self.position[0]+self.radius

	def collide(self, paddle):
		if paddle.color == red:
			if self.left()>=paddle.left() and self.left()<=paddle.right():
				if self.position[1]>=paddle.top() and self.position[1]<=paddle.bottom():
					self.velocity=min((self.velocity+self.velocity*0.20),paddle.thick)
					self.position[0]=paddle.right()+self.radius
					self.direction = self.generate_direction()

					# Red paddle saved ball. Reward!
					paddle.reward += BALL_SAVING_REWARD

		if paddle.color == blue:
			if self.right()<=paddle.right() and self.right()>=paddle.left():
				if self.position[1]>=paddle.top() and self.position[1]<=paddle.bottom():
					self.velocity=min((self.velocity+self.velocity*0.20),paddle.thick)
					self.position[0]=paddle.left()-self.radius
					self.direction = self.generate_direction()

					# Blue paddle saved ball. Reward!
					paddle.reward += BALL_SAVING_REWARD

	def bounceHorizontal(self):
		self.direction=(int)(self.direction)%360
		self.velocity=self.velocity-self.velocity*0.25
		if self.direction<90:
			self.direction=360-self.direction
		elif self.direction<180:
			self.direction=270-(self.direction-90)
		elif self.direction<270:
			self.direction=180-(self.direction-180)
		else:
			self.direction=360-self.direction

	def update(self,paddle1,paddle2):
		# Give reward if paddle gets near to the ball.
		p1_dist = np.linalg.norm(np.array([self.position]) - np.array([[paddle1.column, paddle1.head]]), axis=0)
		p2_dist = np.linalg.norm(np.array([self.position]) - np.array([[paddle2.column, paddle2.head]]), axis=0)

		p1_dist[0] /= width
		p1_dist[1] /= height
		p2_dist[0] /= width
		p2_dist[1] /= height
		p1_dist = np.mean(p1_dist)
		p2_dist = np.mean(p2_dist)

		paddle1.reward += BALL_DISTANCE_REWARD - (p1_dist * BALL_DISTANCE_REWARD)
		paddle2.reward += BALL_DISTANCE_REWARD - (p2_dist * BALL_DISTANCE_REWARD)

		self.position[0]+=self.velocity*math.cos(self.direction*math.pi/180)
		self.position[1]-=self.velocity*math.sin(self.direction*math.pi/180)
		if self.top()<=0:
			self.bounceHorizontal()
			self.position[1]=self.radius
			self.velocity=min((self.velocity+self.velocity*0.20),paddle1.thick)
		elif self.bottom()>=height:
			self.bounceHorizontal()
			self.position[1]=height-self.radius
			self.velocity=min((self.velocity+self.velocity*0.20),paddle1.thick)
		if self.right()<0:
			self.position=[width/2,height/2]
			self.direction = self.generate_direction()
			self.velocity=self.base_velocity

			# Blue scored GOAL!
			paddle2.reward += GOAL_REWARD
			# And Red gets punishment.
			paddle1.reward -= GOAL_PUNISHMENT

			self.last_scorer = 2
			paddle1.head = paddle1.start_pos
			paddle2.head = paddle2.start_pos

		elif self.left()>width:
			self.position=[width/2,height/2]
			self.direction = self.generate_direction()
			self.velocity=self.base_velocity

			# Red scored GOAL!
			paddle1.reward += GOAL_REWARD
			# And Blue gets punishment.
			paddle2.reward -= GOAL_PUNISHMENT

			self.last_scorer = 1

			paddle1.head = paddle1.start_pos
			paddle2.head = paddle2.start_pos

		self.collide(paddle1)
		self.collide(paddle2)

	def reset(self):
		# Ball's direction should be defined randomly for not collecting same experience.
		self.position= [width/2,height/2]
		self.direction = self.generate_direction()

	def generate_direction(self):
		fov = 90
		if random.uniform(0.0, 1.0) > 0.5:
			if random.uniform(0.0, 1.0) > 0.5:
				return random.uniform(0.0, fov/2)
			else:
				return random.uniform(360.0 - fov/2, 360.0)
		else:
			return random.uniform(180.0 - fov/2, 180 + fov/2)

	def draw(self,screen):
		pygame.draw.circle(screen,black,[int(self.position[0]),int(self.position[1])],int(self.radius),0)

class Paddle:
	def __init__(self,_top,_column,_length,_color):
		self.start_pos = _top

		# [x,y] is top left coordinate of paddle
		self.head=_top
		self.column=_column
		self.length=_length
		self.thick=14
		self.color=_color
		self.reward = 0.0

	def top(self):
		return self.head
	def bottom(self):
		return self.top()+self.length
	def left(self):
		return self.column-self.thick/2
	def right(self):
		return self.column+self.thick/2

	def update(self, activity):
		# blocks bar from exceeding screen limit
		if self.head <= 0.0:
			self.head = 0.0
		if self.head+self.length >= height:
			self.head = height-self.length

		# move player bars
		if activity == 0:
			pass
		elif activity == 1:
			self.head -= MOVEMENT_SPEED
		elif activity == 2:
			self.head += MOVEMENT_SPEED
		else:
			raise ValueError("Agent got invalid action value.")

	def reset(self):
		self.head = self.start_pos
		self.reward = 0.0

	def draw(self,screen):
		pygame.draw.line(screen,self.color,[self.column,self.top()],[self.column,self.bottom()],self.thick)

class PyPong:
	def __init__(self, clock_tick):
		self.CLOCK_TICK = clock_tick
		# Initialize the game engine
		pygame.init()
		
		self.size=[width,height]
		self.screen=pygame.display.set_mode(self.size)
		pygame.display.set_caption("PyPong")
		self.clock=pygame.time.Clock()
		
		# Paddle(top,column,length,color)
		self.ball=Ball(width/2,height/2,0)
		self.player1=Paddle(height/3,40,height/6,red)
		self.player2=Paddle(height/3,width-40,height/6,blue)

		# DQN Part.
		self.observation_space_n = len(self.ball.position) + 1
		self.action_space_n = 3		# Nothing, Up, down.

	# State is ball's normalized position relative to screen size.
	def reset(self):
		self.ball.reset()
		self.player1.reset()
		self.player2.reset()

		self.done = False
		return self.get_state()

	def get_state(self):
		r = np.copy(np.array(self.ball.position))
		r[0] /= width
		r[1] /= height

		# Agent's state. Ball + PaddlePos
		return np.squeeze(
			np.array(
				[
					# Agent1 State
					np.array(
						[list(r) + [self.player1.head / height]]
					),

					# Agent2 State
					np.array(
						[list(r) + [self.player2.head / height]]
					),
				]
			)
		)

	def step(self, actions):
		# This limits the while loop to a max of 45 times per second.
		# Leave this out and we will use all CPU we can.
		self.clock.tick(self.CLOCK_TICK)

		self.player1.update(actions[0])
		self.player2.update(actions[1])
		self.ball.update(self.player1,self.player2)
		self.screen.fill(white)

		self.player1.draw(self.screen)
		self.player2.draw(self.screen)
		self.ball.draw(self.screen)

		# Put the image of the text on the screen at 250x250
		font = pygame.font.Font(None, 22)
		fontScore = pygame.font.Font(None, 40)
		self.screen.blit(font.render("PyPong by Beef",True,black), [self.size[0]-140,self.size[1]-30])

		player1_reward = self.player1.reward
		player2_reward = self.player2.reward
		self.player1.reward = 0.0
		self.player2.reward = 0.0

		return self.get_state(), [player1_reward, player2_reward], self.done, None

	def render(self):
		pygame.display.flip()

if __name__ == '__main__':
	env = PyPong()