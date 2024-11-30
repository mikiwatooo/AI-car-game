from pyglet.window import key
from pyglet.gl import *
import pyglet
from Globals import *
import pygame
from Game import Game
import random
import os
import numpy as np
from collections import deque
import tensorflow as tf
import time

tf.compat.v1.disable_eager_execution()

vec2 = pygame.math.Vector2


class QLearning:
    def __init__(self, game):

        self.game = game
        self.game.new_episode()

        self.stateSize = [game.state_size]
        self.actionSize = game.no_of_actions
        self.learningRate = 0.00030
        self.possibleActions = np.identity(self.actionSize, dtype=int)

        self.totalTrainingEpisodes = 100000
        self.maxSteps = 3600

        self.batchSize = 64
        self.memorySize = 100000

        self.maxEpsilon = 1
        self.minEpsilon = 0.01
        self.decayRate = 0.00001
        self.decayStep = 0
        self.gamma = 0.9
        self.training = True

        self.pretrainLength = self.batchSize

        self.maxTau = 10000
        self.tau = 0

        tf.compat.v1.reset_default_graph()

        self.sess = tf.compat.v1.Session()

        self.DQNetwork = DQN(self.stateSize, self.actionSize, self.learningRate, name='DQNetwork')
        self.TargetNetwork = DQN(self.stateSize, self.actionSize, self.learningRate, name='TargetNetwork')

        self.memoryBuffer = PrioritisedMemory(self.memorySize)
        self.pretrain()

        self.state = []
        self.trainingStepNo = 0

        self.newEpisode = False
        self.stepNo = 0
        self.episodeNo = 0
        self.saver = tf.compat.v1.train.Saver()

        load = False
        loadFromEpisodeNo = 15800
        if load:
            self.episodeNo = loadFromEpisodeNo
            self.saver.restore(self.sess, "./allModels/modelMatin{}/models/model.ckpt".format(self.episodeNo))
        else:
            self.sess.run(tf.compat.v1.global_variables_initializer())

        self.sess.run(self.update_target_graph())

    def update_target_graph(self):
        from_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

        to_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

        op_holder = []

        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder

    def pretrain(self):
        for i in range(self.pretrainLength):
            if i == 0:
                state = self.game.get_state()

            action = random.choice(self.possibleActions)
            actionNo = np.argmax(action)
            reward = self.game.make_action(actionNo)
            nextState = self.game.get_state()
            self.newEpisode = False

            if self.game.is_episode_finished():
                reward = -100
                self.memoryBuffer.store((state, action, reward, nextState, True))
                self.game.new_episode()
                state = self.game.get_state()
                self.newEpisode = True
            else:
                self.memoryBuffer.store((state, action, reward, nextState, False))
                self.game.render()
                state = nextState

        print("pretrainingDone")

    def train(self):
        if self.trainingStepNo == 0:
            self.state = self.game.get_state()

        if self.newEpisode:
            self.state = self.game.get_state()

        if self.stepNo < self.maxSteps:
            self.stepNo += 1
            self.decayStep += 1
            self.trainingStepNo += 1
            self.tau += 1

            epsilon = self.minEpsilon + (self.maxEpsilon - self.minEpsilon) * np.exp(
                -self.decayRate * self.decayStep)

            if np.random.rand() < epsilon:
                choice = random.randint(1, len(self.possibleActions)) - 1
                action = self.possibleActions[choice]

            else:
                QValues = self.sess.run(self.DQNetwork.output,
                                        feed_dict={self.DQNetwork.inputs_: np.array([self.state])})
                choice = np.argmax(QValues)
                action = self.possibleActions[choice]

            actionNo = np.argmax(action)
            reward = self.game.make_action(actionNo)

            nextState = self.game.get_state()
            if (reward > 0):
                pass
            
            if self.game.is_episode_finished():
                reward = -100
                self.stepNo = self.maxSteps

            self.memoryBuffer.store((self.state, action, reward, nextState, self.game.is_episode_finished()))

            self.state = nextState

            treeIndexes, batch, ISWeights = self.memoryBuffer.sample(self.batchSize)

            statesFromBatch = np.array([exp[0][0] for exp in batch])
            actionsFromBatch = np.array([exp[0][1] for exp in batch])
            rewardsFromBatch = np.array([exp[0][2] for exp in batch])
            nextStatesFromBatch = np.array([exp[0][3] for exp in batch])
            carDieBooleansFromBatch = np.array([exp[0][4] for exp in batch])

            targetQsFromBatch = []

            QValueOfNextStates = self.sess.run(self.TargetNetwork.output,
                                               feed_dict={self.TargetNetwork.inputs_: nextStatesFromBatch})

            for i in range(self.batchSize):
                action = np.argmax(QValueOfNextStates[i])
                terminalState = carDieBooleansFromBatch[i]
                if terminalState:
                    targetQsFromBatch.append(rewardsFromBatch[i])
                else:

                    target = rewardsFromBatch[i] + self.gamma * QValueOfNextStates[i][action]
                    targetQsFromBatch.append(target)

            targetsForBatch = np.array([t for t in targetQsFromBatch])

            loss, _, absoluteErrors = self.sess.run(
                [self.DQNetwork.loss, self.DQNetwork.optimizer, self.DQNetwork.absoluteError],
                feed_dict={self.DQNetwork.inputs_: statesFromBatch,
                           self.DQNetwork.actions_: actionsFromBatch,
                           self.DQNetwork.targetQ: targetsForBatch,
                           self.DQNetwork.ISWeights_: ISWeights})

            self.memoryBuffer.batchUpdate(treeIndexes, absoluteErrors)

        if self.stepNo >= self.maxSteps:
            self.episodeNo += 1
            self.stepNo = 0
            self.newEpisode = True
            self.game.new_episode()
            if self.episodeNo >= self.totalTrainingEpisodes:
                self.training = False
            if self.episodeNo % 100 == 0:
                directory = "./allModels/model{}".format(self.episodeNo)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_path = self.saver.save(self.sess,
                                            "./allModels/model{}/models/model.ckpt".format(self.episodeNo))
                print("Model Saved")
        if self.tau > self.maxTau:
            self.sess.run(self.update_target_graph())
            self.tau = 0
            print("Target Network Updated")

    def test(self):

        self.state = self.game.get_state()

        QValues = self.sess.run(self.DQNetwork.output,
                                feed_dict={self.DQNetwork.inputs_: np.array([self.state])})
        choice = np.argmax(QValues)
        action = self.possibleActions[choice]

        actionNo = np.argmax(action)

        self.game.make_action(actionNo)

        if self.game.is_episode_finished():
            self.game.new_episode()


class Memory:
    def __init__(self, maxSize):
        self.buffer = deque(maxlen=maxSize)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batchSize):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batchSize,
                                 replace=False)
        return [self.buffer[i] for i in index]


class DQN:
    def __init__(self, stateSize, actionSize, learningRate, name):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.learningRate = learningRate
        self.name = name

        with tf.compat.v1.variable_scope(self.name):
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *self.stateSize], name="inputs")

            self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, self.actionSize], name="actions")

            self.targetQ = tf.compat.v1.placeholder(tf.float32, [None], name="target")

            self.ISWeights_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name='ISWeights')

            self.dense1 = tf.compat.v1.layers.dense(inputs=self.inputs_,
                                          units=16,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                          name="dense1")
            self.dense2 = tf.compat.v1.layers.dense(inputs=self.dense1,
                                          units=16,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                          name="dense2")
            self.output = tf.compat.v1.layers.dense(inputs=self.dense2,
                                          units=self.actionSize,
                                          kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                          activation=None,
                                          name="outputs")

            self.QValue = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            self.absoluteError = abs(self.QValue - self.targetQ)

            self.loss = tf.reduce_mean(self.ISWeights_ * tf.square(self.targetQ - self.QValue))

            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learningRate).minimize(self.loss)


class DDQN:
    def __init__(self, stateSize, actionSize, learningRate, name):
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.learningRate = learningRate
        self.name = name

        with tf.compat.v1.variable_scope(self.name):
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *self.stateSize], name="inputs")

            self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, self.actionSize], name="actions")

            self.targetQ = tf.compat.v1.placeholder(tf.float32, [None], name="target")

            self.ISWeights_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name='ISWeights')

            self.dense1 = tf.compat.v1.layers.dense(inputs=self.inputs_,
                                          units=16,
                                          activation=tf.nn.elu,
                                          kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                          name="dense1")


            self.valueLayer = tf.compat.v1.layers.dense(inputs=self.dense1,
                                              units=16,
                                              activation=tf.nn.elu,
                                              kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                              name="valueLayer")

            self.value = tf.compat.v1.layers.dense(inputs=self.valueLayer,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                         name="value")

            self.advantageLayer = tf.compat.v1.layers.dense(inputs=self.dense1,
                                                  units=16,
                                                  activation=tf.nn.elu,
                                                  kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                                  name="advantageLayer")

            self.advantage = tf.compat.v1.layers.dense(inputs=self.advantageLayer,
                                             units=self.actionSize,
                                             activation=None,
                                             kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                             name="advantages")

            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            self.QValue = tf.reduce_sum(tf.multiply(self.output, self.actions_))

            self.absoluteError = abs(self.QValue - self.targetQ)  # used for prioritising experiences

            self.loss = tf.reduce_mean(self.ISWeights_ * tf.square(self.targetQ - self.QValue))

            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learningRate).minimize(self.loss)


class PrioritisedMemory:
    e = 0.01
    a = 0.06
    b = 0.04
    bIncreaseRate = 0.001
    errorsClippedAt = 1.0

    def __init__(self, capacity):
        self.sumTree = SumTree(capacity)
        self.capacity = capacity

    def store(self, experience):
        maxPriority = np.max(self.sumTree.tree[self.sumTree.indexOfFirstData:])

        if maxPriority == 0:
            maxPriority = self.errorsClippedAt

        self.sumTree.add(maxPriority, experience)

    def sample(self, n):
        batch = []
        batchIndexes = np.zeros([n], dtype=np.int32)
        batchISWeights = np.zeros([n, 1], dtype=np.float32)

        totalPriority = self.sumTree.total_priority()
        prioritySegmentSize = totalPriority / n

        self.b += self.bIncreaseRate
        self.b = min(self.b, 1)

        minPriority = np.min(np.maximum(self.sumTree.tree[self.sumTree.indexOfFirstData:], self.e))
        minProbability = minPriority / self.sumTree.total_priority()

        maxWeight = (minProbability * n) ** (-self.b)
        for i in range(n):
            segmentMin = prioritySegmentSize * i
            segmentMax = segmentMin + prioritySegmentSize

            value = np.random.uniform(segmentMin, segmentMax)

            treeIndex, priority, data = self.sumTree.getLeaf(value)

            samplingProbability = priority / totalPriority

            batchISWeights[i, 0] = np.power(n * samplingProbability, -self.b) / maxWeight

            batchIndexes[i] = treeIndex
            experience = [data]
            batch.append(experience)

        return batchIndexes, batch, batchISWeights

    def batchUpdate(self, treeIndexes, absoluteErrors):
        absoluteErrors += self.e
        clippedErrors = np.minimum(absoluteErrors, self.errorsClippedAt)

        priorities = np.power(clippedErrors, self.a)
        for treeIndex, priority in zip(treeIndexes, priorities):
            self.sumTree.update(treeIndex, priority)


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 2 * capacity - 1
        self.tree = np.zeros(self.size)
        self.data = np.zeros(capacity, dtype=object)
        self.dataPointer = 0
        self.indexOfFirstData = capacity - 1

    def add(self, priority, data):
        treeIndex = self.indexOfFirstData + self.dataPointer

        self.data[self.dataPointer] = data
        self.update(treeIndex, priority)
        self.dataPointer += 1
        self.dataPointer = self.dataPointer % self.capacity

    def update(self, index, priority):
        change = priority - self.tree[index]
        self.tree[index] = priority

        while index != 0:
            index = (index - 1) // 2
            self.tree[index] += change

    def getLeaf(self, value):
        parent = 0
        LChild = 1
        RChild = 2

        while LChild < self.size:
            if self.tree[LChild] >= value:
                parent = LChild
            else:
                value -= self.tree[LChild]
                parent = RChild

            LChild = 2 * parent + 1
            RChild = 2 * parent + 2

        treeIndex = parent
        dataIndex = parent - self.indexOfFirstData

        return treeIndex, self.tree[treeIndex], self.data[dataIndex]

    def total_priority(self):
        return self.tree[0]

class MyWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_minimum_size(400, 300)

        backgroundColor = [0,0,0,1]
        glClearColor(*backgroundColor)

        self.game = Game()
        self.ai = QLearning(self.game)

        self.firstClick = True

    def on_key_press(self, symbol, modifiers):
        pass

    def on_close(self):
        self.ai.sess.close()
        pass

    def on_key_release(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.ai.training = not self.ai.training

    def on_mouse_press(self, x, y, button, modifiers):
        pass

    def on_draw(self):
        window.set_size(width=displayWidth, height=displayHeight)
        self.clear()
        self.game.render()
        vision = self.game.car.getState()
        label=pyglet.text.Label(f"Total time:{self.game.getTotalTime()}\n Attempt time:{int(time.time()-self.game.car.timeAttempt)} sec \nCurrent lap time: {int(time.time()-self.game.car.timeLap)}sec ",y=980,x=20, bold=True)
        label.draw()
        if self.game.car.bestLapTime != 999:
            label=pyglet.text.Label(f"Best lap time:{round(self.game.car.bestLapTime,3)} sec \n ",y=960,x=20, bold=True)
            label.draw()
        

        """ for i in range(len(vision)):

            label = pyglet.text.Label("{}:  {}".format(i,vision[i]),
                                       font_name='Times New Roman',
                                       font_size=12,
                                       x=10, y=20*i+50,
                                       anchor_x='left', anchor_y='center')
            label.draw() """


    def update(self, dt):
        for i in range(5):
            if self.ai.training:
                self.ai.train()
            else:
                self.ai.test()
                return
        pass



if __name__ == "__main__":
    window = MyWindow(displayWidth, displayHeight, "AI Learns to Drive", resizable=False)
    pyglet.clock.schedule_interval(window.update, 1 / frameRate)
    pyglet.app.run()
