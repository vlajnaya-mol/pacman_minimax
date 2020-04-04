# multiAgents.py
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):

    def minimax(self, gameState, depth, playerIndex):
        if playerIndex == gameState.getNumAgents():
            depth -= 1
            playerIndex = 0

        if (depth == 0 or not gameState.getLegalActions(playerIndex)):
            return self.evaluationFunction(gameState), None

        if playerIndex == 0:
            value_actions = [(self.minimax(gameState.generateSuccessor(playerIndex, action),
                                           depth, playerIndex + 1)[0], action)
                             for action in gameState.getLegalActions(playerIndex)]
            return max(value_actions, key=lambda x: x[0])
        else:
            value_actions = [(self.minimax(gameState.generateSuccessor(playerIndex, action),
                                           depth, playerIndex + 1)[0], action)
                             for action in gameState.getLegalActions(playerIndex)]
            return min(value_actions, key=lambda x: x[0])

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, self.depth, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


# def betterEvaluationFunction(currentGameState):
#     def closest_food(cur_pos, food_pos):
#         food_distances = []
#         for food in food_pos:
#             food_distances.append(util.manhattanDistance(food, cur_pos))
#         return min(food_distances) if len(food_distances) > 0 else 1
#
#     def closest_ghost(cur_pos, ghosts):
#         food_distances = []
#         for food in ghosts:
#             food_distances.append(util.manhattanDistance(food.getPosition(), cur_pos))
#         return min(food_distances) if len(food_distances) > 0 else 1
#
#
#     pacman_pos = currentGameState.getPacmanPosition()
#     score = currentGameState.getScore()
#     food = currentGameState.getFood().asList()
#     ghosts = currentGameState.getGhostStates()
#
#
#     cl_dot = closest_food(pacman_pos, food)
#     cl_gh = closest_ghost(pacman_pos, ghosts)
#     score = score * 2 if cl_dot < cl_gh + 3 else score
#
#     return score

def betterEvaluationFunction(currentGameState):
    def closest_dot(cur_pos, food_pos):
        food_distances = []
        for food in food_pos:
            food_distances.append(util.manhattanDistance(food, cur_pos))
        return min(food_distances) if len(food_distances) > 0 else 1

    def closest_ghost(cur_pos, ghosts):
        food_distances = []
        for food in ghosts:
            food_distances.append(util.manhattanDistance(food.getPosition(), cur_pos))
        return min(food_distances) if len(food_distances) > 0 else 1

    def ghost_stuff(cur_pos, ghost_states, radius, scores):
        for ghost in ghost_states:
            gdist = util.manhattanDistance(ghost.getPosition(), cur_pos)
            if gdist <= radius:
                scores -= 9 / (1 + gdist)
        return scores

    def food_stuff(cur_pos, food_positions):
        food_distances = []
        for food in food_positions:
            dist = util.manhattanDistance(food, cur_pos)
            if dist < 8:
                if dist < 4:
                    dist /= 1.2
                dist /= 1.1
            if dist > 15:
                dist /= 2
            food_distances.append(dist)
        return sum(food_distances)


    pacman_pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()

    score = score * 2 if closest_dot(pacman_pos, food) < closest_ghost(pacman_pos, ghosts) + 3 else score
    score -= food_stuff(pacman_pos, food) / 20
    score = ghost_stuff(pacman_pos, ghosts, 8, score)
    return score


# Abbreviation
better = betterEvaluationFunction
