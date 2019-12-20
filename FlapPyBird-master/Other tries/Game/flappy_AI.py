from itertools import cycle
import random
import sys
import numpy as np

import pygame

#from AI_models import GeneticAi
from Ai_models import GeneticAi_torch

from pygame.locals import *

FPS = 100
SCREENWIDTH = 288
SCREENHEIGHT = 512

max_score_over_gen = 0
best_model_over_gen = None

n_generations = 0
N_POPULATION = 50

PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

playerWeigths = GeneticAi_torch.initialize(N_POPULATION,2)
playerLast3Scores = []
for i in range(N_POPULATION):
  playerLast3Scores.append(np.zeros((1,3)))

# Object player to create multples instances
class Player:

  def __init__(self, playerx, playery, playerIndex, playerVelY, playerMaxVelY, playerMinVelY, playerAccY, playerRot, playerVelRot, playerRotThr,
               playerFlapAcc, playerFlapped, playerScore, playerScore_i, playerWeigths, playerLast3Scores):
    self.playerx = playerx
    self.playery = playery
    self.playerScore_i = playerScore_i
    self.playerLast3Scores = playerLast3Scores
    self.meanScore_last3 = np.sum(playerLast3Scores)/3
    self.playerIndex = playerIndex
    self.playerVelY = playerVelY  # player's velocity along Y, default same as playerFlapped
    self.playerMaxVelY = playerMaxVelY  # max vel along Y, max descend speed
    self.playerMinVelY = playerMinVelY  # min vel along Y, max ascend speed
    self.playerAccY = playerAccY  # players downward accleration
    self.playerRot = playerRot  # player's rotation
    self.playerVelRot = playerVelRot  # angular speed
    self.playerRotThr = playerRotThr  # rotation threshold
    self.playerFlapAcc = playerFlapAcc  # players speed on flapping
    self.playerFlapped = playerFlapped  # True when player flaps
    self.playerScore = playerScore
    self.playerWeights = playerWeigths


# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
  # red bird
  (
    'assets/sprites/redbird-upflap.png',
    'assets/sprites/redbird-midflap.png',
    'assets/sprites/redbird-downflap.png',
  ),
  # blue bird
  (
    'assets/sprites/bluebird-upflap.png',
    'assets/sprites/bluebird-midflap.png',
    'assets/sprites/bluebird-downflap.png',
  ),
  # yellow bird
  (
    'assets/sprites/yellowbird-upflap.png',
    'assets/sprites/yellowbird-midflap.png',
    'assets/sprites/yellowbird-downflap.png',
  ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
  'assets/sprites/background-day.png',
  'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
  'assets/sprites/pipe-green.png',
  'assets/sprites/pipe-red.png',
)

try:
  xrange
except NameError:
  xrange = range


def main():
  global SCREEN, FPSCLOCK
  pygame.init()
  FPSCLOCK = pygame.time.Clock()
  SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
  pygame.display.set_caption('Flappy Bird')

  # numbers sprites for score display
  IMAGES['numbers'] = (
    pygame.image.load('assets/sprites/0.png').convert_alpha(),
    pygame.image.load('assets/sprites/1.png').convert_alpha(),
    pygame.image.load('assets/sprites/2.png').convert_alpha(),
    pygame.image.load('assets/sprites/3.png').convert_alpha(),
    pygame.image.load('assets/sprites/4.png').convert_alpha(),
    pygame.image.load('assets/sprites/5.png').convert_alpha(),
    pygame.image.load('assets/sprites/6.png').convert_alpha(),
    pygame.image.load('assets/sprites/7.png').convert_alpha(),
    pygame.image.load('assets/sprites/8.png').convert_alpha(),
    pygame.image.load('assets/sprites/9.png').convert_alpha()
  )

  # game over sprite
  IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
  # message sprite for welcome screen
  IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
  # base (ground) sprite
  IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

  # sounds
  if 'win' in sys.platform:
    soundExt = '.wav'
  else:
    soundExt = '.ogg'

  SOUNDS['die'] = pygame.mixer.Sound('assets/audio/die' + soundExt)
  SOUNDS['hit'] = pygame.mixer.Sound('assets/audio/hit' + soundExt)
  SOUNDS['point'] = pygame.mixer.Sound('assets/audio/point' + soundExt)
  SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
  SOUNDS['wing'] = pygame.mixer.Sound('assets/audio/wing' + soundExt)

  while True:
    # select random background sprites
    randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

    # select random player sprites
    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
      pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
      pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
      pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    # select random pipe sprites
    pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES['pipe'] = (
      pygame.transform.flip(
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
      pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
      getHitmask(IMAGES['pipe'][0]),
      getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
      getHitmask(IMAGES['player'][0]),
      getHitmask(IMAGES['player'][1]),
      getHitmask(IMAGES['player'][2]),
    )

    movementInfo = showWelcomeAnimation()
    crashInfo = mainGame(movementInfo)
    showGameOverScreen(crashInfo)

def showAI_info(i,alive,total_population, max_score_over_gen, n_generations):
  black = (0,0,0)
  myFont = pygame.font.SysFont("Times New Roman", 18)

  randNumLabel = myFont.render("Score: ", 1, black)
  ### pass a string to myFont.render
  diceDisplay = myFont.render(str(i), 1, black)

  SCREEN.blit(randNumLabel, (10, 10))
  SCREEN.blit(diceDisplay, (60, 10))

  randNumLabel = myFont.render("Alive: ", 1, black)
  ### pass a string to myFont.render
  diceDisplay = myFont.render(str(alive), 1, black)

  SCREEN.blit(randNumLabel, (10, 40))
  SCREEN.blit(diceDisplay, (60, 40))

  randNumLabel = myFont.render("/", 1, black)
  diceDisplay = myFont.render(str(total_population), 1, black)

  SCREEN.blit(randNumLabel, (80, 40))
  SCREEN.blit(diceDisplay, (100, 40))

  randNumLabel = myFont.render("MAX_SCORE: ", 1, black)
  diceDisplay = myFont.render(str(max_score_over_gen), 1, black)

  SCREEN.blit(randNumLabel, (130, 10))
  SCREEN.blit(diceDisplay, (250, 10))

  randNumLabel = myFont.render("Generacion: ", 1, black)
  diceDisplay = myFont.render(str(n_generations), 1, black)

  SCREEN.blit(randNumLabel, (10,70))
  SCREEN.blit(diceDisplay, (100, 70))

  return




def showWelcomeAnimation():
  """Shows welcome screen animation of flappy bird"""
  # index of player to blit on screen
  playerIndex = 0
  playerIndexGen = cycle([0, 1, 2, 1])
  # iterator used to change playerIndex after every 5th iteration
  loopIter = 0

  playerx = int(SCREENWIDTH * 0.2)
  playery = int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2)

  messagex = int((SCREENWIDTH - IMAGES['message'].get_width()) / 2)
  messagey = int(SCREENHEIGHT * 0.12)

  basex = 0
  # amount by which base can maximum shift to left
  baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

  # player shm for up-down motion on welcome screen
  playerShmVals = {'val': 0, 'dir': 1}

  while True:
    for event in pygame.event.get():
      if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
        pygame.quit()
        sys.exit()
      if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
        # make first flap sound and return values for mainGame
        SOUNDS['wing'].play()
        return {
          'playery': playery + playerShmVals['val'],
          'basex': basex,
          'playerIndexGen': playerIndexGen,
        }

    # adjust playery, playerIndex, basex
    if (loopIter + 1) % 5 == 0:
      playerIndex = next(playerIndexGen)
    loopIter = (loopIter + 1) % 30
    basex = -((-basex + 4) % baseShift)
    playerShm(playerShmVals)

    # draw sprites
    SCREEN.blit(IMAGES['background'], (0, 0))
    SCREEN.blit(IMAGES['player'][playerIndex],
                (playerx, playery + playerShmVals['val']))
    SCREEN.blit(IMAGES['message'], (messagex, messagey))
    SCREEN.blit(IMAGES['base'], (basex, BASEY))

    pygame.display.update()
    FPSCLOCK.tick(FPS)


def mainGame(movementInfo):
  global max_score_over_gen
  global n_generations
  global playerWeigths
  global best_model_over_gen
  global playerLast3Scores

  score = playerIndex = loopIter = 0
  playerIndexGen = movementInfo['playerIndexGen']
  playerx, playery = int(SCREENWIDTH * 0.2), movementInfo['playery']

  basex = movementInfo['basex']
  baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

  # get 2 new pipes to add to upperPipes lowerPipes list
  newPipe1 = getRandomPipe()
  newPipe2 = getRandomPipe()

  # list of upper pipes
  upperPipes = [
    {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
    {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
  ]

  # list of lowerpipe
  lowerPipes = [
    {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
    {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
  ]

  pipeVelX = -4

  # player velocity, max velocity, downward accleration, accleration on flap
  playerVelY = -9  # player's velocity along Y, default same as playerFlapped
  playerMaxVelY = 10  # max vel along Y, max descend speed
  playerMinVelY = -8  # min vel along Y, max ascend speed
  playerAccY = 1  # players downward accleration
  playerRot = 45  # player's rotation
  playerVelRot = 3  # angular speed
  playerRotThr = 20  # rotation threshold
  playerFlapAcc = -9  # players speed on flapping
  playerFlapped = False  # True when player flaps


  players = []


  for i in range(N_POPULATION):
    players.append(
      Player(playerx,playery,playerIndex,playerVelY, playerMaxVelY, playerMinVelY, playerAccY, playerRot, playerVelRot, playerRotThr, playerFlapAcc,
             playerFlapped, 0,0,playerWeigths[i], playerLast3Scores[i]))
  i = 0
  crash_list = [False]*len(players)


  pipe2pass = lowerPipes[0]
  future_pipe2pass = pipe2pass

  count_pipe2pass = 0

  while True:

    pygame.event.get()

    # COMPROBAMOS QUE NO HA CAMBIADO LA TUBERIA A LA QUE TENEMOS QUE APUNTAR
    count_pipe2pass += 1

    if count_pipe2pass > 4:
      if pipe2pass != future_pipe2pass:
        pipe2pass = future_pipe2pass

    if i == 0:
      for event in pygame.event.get():
        if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
          pygame.quit()
          sys.exit()
        if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
          if playery > -2 * IMAGES['player'][0].get_height():
            playerVelY = playerFlapAcc
            playerFlapped = True
            SOUNDS['wing'].play()
    elif i%1 == 0:
      #choices = np.random.choice([0, 1], size=(len(players),))
      for id_player in range(0,N_POPULATION):
        if not crash_list[id_player]:


          pipeMidPos_y = (pipe2pass['y'] + 50) / SCREENHEIGHT

          player_y_env = players[id_player].playery/380


          #env = [pipeVelX_env, pipeMidPos_env, player_y_env,dist_x_env]
          env = [player_y_env, pipeMidPos_y]
          #env = [player_y_env, pipeMidPos_y,dist_x_env]

          choice = GeneticAi_torch.predict(playerWeigths[id_player], env)

          if choice == 1 and playery > -2 * IMAGES['player'][0].get_height():
            players[id_player].playerVelY = players[id_player].playerFlapAcc
            players[id_player].playerFlapped = True
            SOUNDS['wing'].play()

    else:
      pass


    # check for crash here
    for j in range(len(crash_list)):
      if not crash_list[j]:
        crash_list[j] = checkCrash({'x': players[j].playerx, 'y': players[j].playery, 'index': players[j].playerIndex},
                           upperPipes, lowerPipes)[0]

        if crash_list[j] == True:
          players[j].playerLast3Scores = [np.concatenate(([players[j].playerScore_i],players[j].playerLast3Scores[0][0:2]))]
          players[j].meanScore_last3 = np.sum(players[j].playerLast3Scores)/3
          #players[j].playerScore_i = player.playerScore_i - 100

    alive = len([k for k, x in enumerate(crash_list) if x == False])
    total_population = len(crash_list)

    if all(crash_list):
      max_score = 0
      max_score_index = 0

      players.sort(key=lambda x: x.playerScore_i, reverse=True)
      #players.sort(key=lambda x: x.meanScore_last3, reverse=True)

      max_score = players[0].meanScore_last3
      playerWeigths = []
      playerLast3Scores = []

      for index,player in enumerate(players):
        playerWeigths.append(player.playerWeights)
        playerLast3Scores.append(player.playerLast3Scores)
        if index == 0:
          print(player.playerScore_i)
          print()
          pass

      playerWeigths = GeneticAi_torch.train(playerWeigths)
      #if best_model_over_gen is not None:
      #  playerWeigths[-3] = best_model_over_gen
      n_generations +=1
      mainGame(movementInfo)

      return {
        'y': players[max_score_index].playery,
        'groundCrash': True,
        'basex': basex,
        'upperPipes': upperPipes,
        'lowerPipes': lowerPipes,
        'score': players[max_score_index].playerScore,
        'playerVelY': players[max_score_index].playerVelY,
        'playerRot': players[max_score_index].playerRot
      }

    # check for score
    for j,player in enumerate(players):
      if crash_list[j]:
        continue
      playerMidPos = player.playerx + IMAGES['player'][0].get_width() / 2
      player.playerScore_i = player.playerScore_i  + 1
      for pipe in upperPipes:
        pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
        if pipeMidPos <= playerMidPos < pipeMidPos + 4:
          player.playerScore += 1
          player.playerScore_i = player.playerScore_i + 100
          count_pipe2pass = 0

          for pipe in lowerPipes:
            if pipe2pass['y'] != pipe['y']:
              future_pipe2pass = pipe
              break

          SOUNDS['point'].play()

    # playerIndex basex change
    for j,player in enumerate(players):
      if crash_list[j]:
        continue
      if (loopIter + 1) % 3 == 0:
        player.playerIndex = next(playerIndexGen)
      loopIter = (loopIter + 1) % 30
      basex = -((-basex + 100) % baseShift)

    # rotate the player
    for j,player in enumerate(players):
      if crash_list[j]:
        continue
      if player.playerRot > -90:
        player.playerRot -= player.playerVelRot

    # player's movement
    for j,player in enumerate(players):
      if crash_list[j]:
        continue
      if player.playerVelY < player.playerMaxVelY and not player.playerFlapped:
        player.playerVelY += player.playerAccY
      if player.playerFlapped:
        player.playerFlapped = False
        player.playerRot = 45

      # more rotation to cover the threshold (calculated in visible rotation)

    for j,player in enumerate(players):
      if crash_list[j]:
        continue
      player.playerHeight = IMAGES['player'][player.playerIndex].get_height()
      player.playery += min(player.playerVelY, BASEY - player.playery - player.playerHeight)
      if player.playery <= 0.0:
          #player.playery = max(0,player.playery)
          crash_list[j] = True

    # move pipes to left
    for uPipe, lPipe in zip(upperPipes, lowerPipes):
      uPipe['x'] += pipeVelX
      lPipe['x'] += pipeVelX

    # add new pipe when first pipe is about to touch left of screen

    if 0 < upperPipes[0]['x'] < 5:
      newPipe = getRandomPipe()
      upperPipes.append(newPipe[0])
      lowerPipes.append(newPipe[1])


    # remove first pipe if its out of the screen
    if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
      upperPipes.pop(0)
      lowerPipes.pop(0)

    # draw sprites
    SCREEN.blit(IMAGES['background'], (0, 0))

    for uPipe, lPipe in zip(upperPipes, lowerPipes):
      SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
      SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

    SCREEN.blit(IMAGES['base'], (basex, BASEY))
    # print score so player overlaps the score

    best_player_index = 0
    max_score = 0
    for j,player in enumerate(players):
      if player.playerScore_i > max_score_over_gen:
        max_score_over_gen = player.playerScore_i
        best_model_over_gen = player.playerWeights

      if player.playerScore > max_score:
        best_player_index = j
        max_score = player.playerScore



    showScore(players[best_player_index].playerScore)
    showAI_info(i,alive,total_population,max_score_over_gen, n_generations)
    # Player rotation has a threshold
    for j,player in enumerate(players):
      if crash_list[j]:
        continue
      visibleRot = player.playerRotThr
      if player.playerRot <= player.playerRotThr:
        visibleRot = player.playerRot

      playerSurface = pygame.transform.rotate(IMAGES['player'][player.playerIndex], visibleRot)
      SCREEN.blit(playerSurface, (player.playerx, player.playery))

    pygame.display.update()
    i+=1
    #FPSCLOCK.tick(5)
    FPSCLOCK.tick(FPS)

def showGameOverScreen(crashInfo):
  """crashes the player down ans shows gameover image"""
  score = crashInfo['score']
  playerx = SCREENWIDTH * 0.2
  playery = crashInfo['y']
  playerHeight = IMAGES['player'][0].get_height()
  playerVelY = crashInfo['playerVelY']
  playerAccY = 2
  playerRot = crashInfo['playerRot']
  playerVelRot = 7

  basex = crashInfo['basex']

  upperPipes, lowerPipes = crashInfo['upperPipes'], crashInfo['lowerPipes']

  # play hit and die sounds
  SOUNDS['hit'].play()
  if not crashInfo['groundCrash']:
    SOUNDS['die'].play()

  while True:
    for event in pygame.event.get():
      if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
        pygame.quit()
        sys.exit()
      if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
        if playery + playerHeight >= BASEY - 1:
          return

    # player y shift
    if playery + playerHeight < BASEY - 1:
      playery += min(playerVelY, BASEY - playery - playerHeight)

    # player velocity change
    if playerVelY < 15:
      playerVelY += playerAccY

    # rotate only when it's a pipe crash
    if not crashInfo['groundCrash']:
      if playerRot > -90:
        playerRot -= playerVelRot

    # draw sprites
    SCREEN.blit(IMAGES['background'], (0, 0))

    for uPipe, lPipe in zip(upperPipes, lowerPipes):
      SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
      SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

    SCREEN.blit(IMAGES['base'], (basex, BASEY))
    showScore(score)

    playerSurface = pygame.transform.rotate(IMAGES['player'][1], playerRot)
    SCREEN.blit(playerSurface, (playerx, playery))
    SCREEN.blit(IMAGES['gameover'], (50, 180))

    FPSCLOCK.tick(FPS)
    pygame.display.update()


def playerShm(playerShm):
  """oscillates the value of playerShm['val'] between 8 and -8"""
  if abs(playerShm['val']) == 8:
    playerShm['dir'] *= -1

  if playerShm['dir'] == 1:
    playerShm['val'] += 1
  else:
    playerShm['val'] -= 1


def getRandomPipe():
  """returns a randomly generated pipe"""
  # y of gap between upper and lower pipe
  gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
  gapY += int(BASEY * 0.2)
  pipeHeight = IMAGES['pipe'][0].get_height()
  pipeX = SCREENWIDTH + 10

  return [
    {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
    {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
  ]


def showScore(score):
  """displays score in center of screen"""
  scoreDigits = [int(x) for x in list(str(score))]
  totalWidth = 0  # total width of all numbers to be printed

  for digit in scoreDigits:
    totalWidth += IMAGES['numbers'][digit].get_width()

  Xoffset = (SCREENWIDTH - totalWidth) / 2

  for digit in scoreDigits:
    SCREEN.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
    Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
  """returns True if player collders with base or pipes."""
  pi = player['index']
  player['w'] = IMAGES['player'][0].get_width()
  player['h'] = IMAGES['player'][0].get_height()

  # if player crashes into ground
  if player['y'] + player['h'] >= BASEY - 1:
    return [True, True]
  else:

    playerRect = pygame.Rect(player['x'], player['y'],
                             player['w'], player['h'])
    pipeW = IMAGES['pipe'][0].get_width()
    pipeH = IMAGES['pipe'][0].get_height()

    if len(lowerPipes) > 2:
      print()

    for uPipe, lPipe in zip(upperPipes, lowerPipes):
      # upper and lower pipe rects

      uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
      lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

      # player and upper/lower pipe hitmasks
      pHitMask = HITMASKS['player'][pi]
      uHitmask = HITMASKS['pipe'][0]
      lHitmask = HITMASKS['pipe'][1]

      # if bird collided with upipe or lpipe
      uCollide = pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
      lCollide = pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

      if uCollide or lCollide:
        return [True, False]

  return [False, False]


def pixelCollision(rect1, rect2, hitmask1, hitmask2):
  """Checks if two objects collide and not just their rects"""
  rect = rect1.clip(rect2)

  if rect.width == 0 or rect.height == 0:
    return False

  x1, y1 = rect.x - rect1.x, rect.y - rect1.y
  x2, y2 = rect.x - rect2.x, rect.y - rect2.y

  for x in xrange(rect.width):
    for y in xrange(rect.height):
      if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
        return True
  return False


def getHitmask(image):
  """returns a hitmask using an image's alpha."""
  mask = []
  for x in xrange(image.get_width()):
    mask.append([])
    for y in xrange(image.get_height()):
      mask[x].append(bool(image.get_at((x, y))[3]))
  return mask


if __name__ == '__main__':
  main()
  print("TEST COMMIT")
