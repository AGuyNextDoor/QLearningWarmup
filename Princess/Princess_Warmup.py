import pprint
import os
import numpy as np
import re
import math

from printQTable import printQTable
from checkEquality import checkEquality

clear = lambda: os.system('clear')
#################################################################################################################
n = 5
boardPositionValue = np.array([[0.04 for j in range(n)] for i in range(n)])
boardPositionValue[0,0],boardPositionValue[1,1],boardPositionValue[1,3],boardPositionValue[3,1],boardPositionValue[3,3],boardPositionValue[4,2] = (0, -1, -1, -1, -1, 1);

qtable = np.array([[0.00 for j in range(10)] for i in range(10)])
# qtable[2, 4] = 2;
# qtable[2, 3] = 6;
# qtable[2, 2] = 7;
# qtable[5,7] = 3;

initialState = [0,0]
episode = 1
episodes = 1
totalEpisodes = 1000
alpha = 0.2
epsilon = 1
epsilon_rate = 0.001
gamma = 0.9

#################################################################################################################
#Utility functions

def possibleMove(currentState):
  x, y = currentState
  possibleMoves = ["L", "R", "U", "D"]
  if(y == 0):
    possibleMoves.remove("L")
  if(y == 4):
    possibleMoves.remove("R")
  if(x == 0):
    possibleMoves.remove("U")
  if(x == 4):
    possibleMoves.remove("D")

  return possibleMoves

def qtableValues(currentState, possibleMoves):
  x, y = currentState
  qX = x*2
  qY = y*2
  facingRewards = []

  for m in possibleMoves:
    if(m == "L"):
      facingRewards.append(qtable[qX,qY])
    if(m == "R"):
      facingRewards.append(qtable[qX,qY+1])
    if(m == "U"):
      facingRewards.append(qtable[qX+1,qY])
    if(m == "D"):
      facingRewards.append(qtable[qX+1,qY+1])

  return facingRewards

def ploitOrPlor(currentState, possibleMoves, facingRewards):
  wall = np.random.uniform(0, 1)
  if(checkEquality(facingRewards)):
    wall = -10
  print(wall)
  if(wall < epsilon):                                               #exploration
    denominator = len(possibleMoves)
    # pathChosen = math.ceil(np.random.randint(0,1)*3) - 1
    pathChosen = np.random.randint(1,denominator+1) - 1
    # print("path", pathChosen)


  else :                                                           #exploitation
    # print("look at the moves : ",possibleMoves)
    # print("rewards list : ", facingRewards)
    pathChosen = facingRewards.index(max(facingRewards))

  xQPath = currentState[0]*2
  yQPath = currentState[1]*2
  if(possibleMoves[pathChosen] == "U" or possibleMoves[pathChosen] == "D"):
    # print("been here")
    xQPath += 1
  if(possibleMoves[pathChosen] == "R" or possibleMoves[pathChosen] == "D"):
    # print("been there")
    yQPath += 1
  # print("original board position :", currentState, " -> ", possibleMoves[pathChosen])
  # print("QPath chosed", [xQPath, yQPath])
  path = [xQPath, yQPath]
  return path, possibleMoves[pathChosen], facingRewards[pathChosen]


def nextAction(currentState, epsilon):
  possibleMoves = possibleMove(currentState)
  # print(possibleMoves)
  facingRewards = qtableValues(currentState, possibleMoves)
  # print(facingRewards)
  qPathChosen, actionLetter, consequentReward = ploitOrPlor(currentState, possibleMoves, facingRewards)
  return qPathChosen, possibleMoves, facingRewards, actionLetter, consequentReward

def newPosition(currentState, actionLetter):
  newPosition = currentState
  if(actionLetter == "L"):
    newPosition = [newPosition[0], newPosition[1]-1]
  if(actionLetter == "R"):
    newPosition = [newPosition[0], newPosition[1]+1]
  if(actionLetter == "U"):
    newPosition = [newPosition[0]-1, newPosition[1]]
  if(actionLetter == "D"):
    newPosition = [newPosition[0]+1, newPosition[1]]

  return newPosition



#################################################################################################################
#FIle execution
deaths = 0
deathStep = []
wins = 0
winStep = []
# deathsAndWins = []
steps = 1
flag = True

while(episode < totalEpisodes):
  print("episode is %s" % episode)
  while(flag):
    if(episodes == 1): currentBoardPosition = initialState
    # clear()
    # print(boardPositionValue, "\n\n")
    # printQTable(qtable)
    print("\n")

    # if(episode == 25): printQTable(qtable)
    # if(episode == 50): printQTable(qtable)
    # if(episode == 100): printQTable(qtable)
    # if(episode == 200): printQTable(qtable)
    # if(episode == 300): printQTable(qtable)
    # if(episode == 400): printQTable(qtable)
    # if(episode == 600): printQTable(qtable)
    # if(episode == 800): printQTable(qtable)
    # if(episode == 900): printQTable(qtable)
    if(episode == 999): printQTable(qtable)

    qAction, possibleMoves, facingRewards, actionLetter, consequentReward = nextAction(currentBoardPosition, epsilon)
    newBoardPosition = newPosition(currentBoardPosition, actionLetter)
    qActionPrime, possibleMovesPrime, facingRewardsPrime, actionLetterPrime, consequentRewardPrime = nextAction(newBoardPosition, -1000)

    # print("qaction : ", qAction)
    # print("actionLetter : ", actionLetter)
    # print("consequentReward : ", consequentReward)
    # print("possibleMoves : ", possibleMoves)
    # print("facingRewards : ", facingRewards)
    # print("newBoardPosition : ", newBoardPosition)
    # print("boardPositionValue[newBoardPosition[0], newBoardPosition[1]] : ", boardPositionValue[newBoardPosition[0], newBoardPosition[1]])

    # print(nextAction(newBoardPosition, -1000))
    deltaQ = boardPositionValue[newBoardPosition[0], newBoardPosition[1]] + (gamma*(consequentRewardPrime)) - consequentReward
    newQ = consequentReward + alpha*deltaQ
    # print("deltaQ value : ", deltaQ)
    # print("newQ value : ", newQ)
    # print(qAction[0], qAction[1])
    # print(qtable[qAction[0], qAction[1]])
    qtable[qAction[0], qAction[1]] = newQ
    # qtable[2,4] = newQ
    # print(qtable[qAction[0], qAction[1]])
    if(epsilon < 0.05): epsilon = 0.05
    else: epsilon -= epsilon_rate

    # printQTable(qtable)
    print("\n")
    steps += 1
    if(newBoardPosition == [1,1] or newBoardPosition == [1,3] or newBoardPosition == [3,1] or newBoardPosition == [3,3]):
      currentBoardPosition = [0,0]
      deaths += 1
      deathStep.append([steps])
      steps = 0
      flag = False
    elif(newBoardPosition == [4,2]):
      currentBoardPosition = [0,0]
      wins += 1
      winStep.append([steps])
      steps = 0
      flag = False
    else:
      currentBoardPosition = newBoardPosition
      print(currentBoardPosition)
    episodes += 1
  episode = episode + 1
  flag = True
  print("episode number is : %s out of %s episodes and epsilon is %.3f" % (episode, totalEpisodes, epsilon))


# input("Press Enter to continue...")
# os.execl(sys.executable, sys.executable, *sys.argv)
print("total number of deaths : %s\n" % deaths)
# print(deathStep)
print("average number of steps before death : %.3f\n" % np.mean(deathStep))

print("total number of wins : %s\n" % wins)
# print(winStep)
print("average number of steps before win : %.3f\n" % np.mean(winStep))
print("average number of steps for last 300 win : %.3f\n" % np.mean(winStep[:-300]))
printQTable(qtable)
