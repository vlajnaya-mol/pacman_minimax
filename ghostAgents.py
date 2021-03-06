# ghostAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import Agent
from game import Actions
import random
from game import Directions
import random
from util import manhattanDistance
import util


class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index): dist[a] = 1.0
        dist.normalize()
        return dist


def ghostEvanuationFunction(currentGameState):
    if currentGameState.isLose():
        return 1
    return -sum([manhattanDistance(currentGameState.getPacmanPosition(), ghost.getPosition())
                for ghost in currentGameState.getGhostStates()])


class MinimaxGhost(GhostAgent):
    def __init__(self, index, depth=3):
        GhostAgent.__init__(self, index=index)
        self.depth = depth

    def getAction(self, state):
        if random.random() > 0.90:
            return random.choice(state.getLegalActions(self.index))
        return self.minimax(state, self.depth, self.index)[1]

    def minimax(self, gameState, depth, playerIndex):
        if playerIndex == gameState.getNumAgents():
            depth -= 1
            playerIndex = 0

        if ((depth == 0 and playerIndex == self.index) or not gameState.getLegalActions(playerIndex)):
            return ghostEvanuationFunction(gameState), None

        if playerIndex == 0:
            value_actions = [(self.minimax(gameState.generateSuccessor(playerIndex, action),
                                           depth, playerIndex + 1)[0], action)
                             for action in gameState.getLegalActions(playerIndex)]
            return min(value_actions, key=lambda x: x[0])
        else:
            value_actions = [(self.minimax(gameState.generateSuccessor(playerIndex, action),
                                           depth, playerIndex + 1)[0], action)
                             for action in gameState.getLegalActions(playerIndex)]
            return max(value_actions, key=lambda x: x[0])

    def getDistribution(self, state):
        raise NotImplementedError


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector(a, speed) for a in legalActions]
        newPositions = [(pos[0] + a[0], pos[1] + a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += (1 - bestProb) / len(legalActions)
        dist.normalize()
        return dist
