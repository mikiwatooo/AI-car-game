import numpy as np
from Globals import *
from Drawer import Drawer
from ShapeObjects import *
from PygameAdditionalMethods import *
import pygame
import time

drawer = Drawer()
vec2 = pygame.math.Vector2

class Game:
    no_of_actions = 9
    state_size = 15

    def __init__(self):
        trackImg = pyglet.image.load('./Data/'+tracks[trackNo]+'.png')
        self.trackSprite = pyglet.sprite.Sprite(trackImg, x=0, y=0)


        # initiate walls
        self.walls = []
        self.gates = []

        self.set_walls()
        self.set_gates()
        self.firstClick = True

        self.car = Car(self.walls, self.gates)

    def set_walls(self):
        with open("./Data/"+tracks[trackNo]+"_walls.txt","r") as f:
            lines=f.readlines()
            for line in lines:
                self.walls.append(Wall(*[int(num) for num in line.split(",")]))


    def set_gates(self):
        with open("./Data/"+tracks[trackNo]+"_gates.txt","r") as f:
            lines=f.readlines()
            for line in lines:
                self.gates.append(RewardGate(*[int(num) for num in line.split(",")]))

    def new_episode(self):
        self.car.reset()

    def get_state(self):
        return self.car.getState()

    def make_action(self, action):
        # returns reward
        actionNo = np.argmax(action)
        self.car.updateWithAction(actionNo)
        return self.car.reward

    def is_episode_finished(self):
        return self.car.dead

    def get_score(self):
        return self.car.score

    def get_lifespan(self):
        return self.car.lifespan

    def render(self):
        glPushMatrix()
        self.trackSprite.draw()

        for w in self.walls:
            w.draw()
        for g in self.gates:
            g.draw()
        self.car.update()
        self.car.show()
        self.car.showCollisionVectors()

        glPopMatrix()

    def getTotalTime(self):
        sec=int(time.time()-self.car.timeTotal)
        if sec<60:
            return f"{sec}s"
        elif sec<3600:
            return f"{sec//60}m:{sec%60}s"
        else:
            return f"{sec//3600}h:{sec//60}m:{sec%60}"



class Wall:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = displayHeight - y1
        self.x2 = x2
        self.y2 = displayHeight - y2

        self.line = Line(self.x1, self.y1, self.x2, self.y2)
        self.line.setLineThinkness(5)

    """
    draw the line
    """
    def draw(self):
        self.line.draw()
    """
    returns true if the car object has hit this wall
    """

    def hitCar(self, car):
        global vec2
        cw = car.width
        ch = car.height
        rightVector = vec2(car.direction)
        upVector = vec2(car.direction).rotate(-90)
        carCorners = []
        cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        carPos = vec2(car.x, car.y)
        for i in range(4):
            carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
                              (upVector * ch / 2 * cornerMultipliers[i][1]))

        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
                             carCorners[j].y):
                return True
        return False


"""
class containing all the game logic for moving and displaying the car
"""


class RewardGate:

    def __init__(self, x1, y1, x2, y2):
        global vec2
        self.x1 = x1
        self.y1 = displayHeight - y1
        self.x2 = x2
        self.y2 = displayHeight - y2
        self.active = True

        self.line = Line(self.x1, self.y1, self.x2, self.y2)
        self.line.setLineThinkness(5)
        self.line.setColor([0, 255, 0])

        self.center = vec2((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    """
    draw the line
    """

    def draw(self):
        if self.active:
            self.line.draw()

    """
    returns true if the car object has hit this wall
    """

    def hitCar(self, car):

        if not self.active:
            return False

        global vec2
        cw = car.width
        ch = car.height
        rightVector = vec2(car.direction)
        upVector = vec2(car.direction).rotate(-90)
        carCorners = []
        cornerMultipliers = [[1, 1], [1, -1], [-1, -1], [-1, 1]]
        carPos = vec2(car.x, car.y)
        for i in range(4):
            carCorners.append(carPos + (rightVector * cw / 2 * cornerMultipliers[i][0]) +
                              (upVector * ch / 2 * cornerMultipliers[i][1]))

        for i in range(4):
            j = i + 1
            j = j % 4
            if linesCollided(self.x1, self.y1, self.x2, self.y2, carCorners[i].x, carCorners[i].y, carCorners[j].x,
                             carCorners[j].y):
                return True
        return False





class Car:

    def __init__(self, walls, rewardGates):
        global vec2


        self.timeTotal=time.time()
        self.timeAttempt=time.time()
        self.timeLap=time.time()
        self.bestLapTime=999

        self.nbVect = 16
        self.angles = a=np.append(np.append(np.linspace(-135,-45,self.nbVect//4),np.linspace(-35,35,self.nbVect//2)),np.linspace(45,135,self.nbVect//4))
        self.x = startingPositinos[trackNo][0]
        self.y = startingPositinos[trackNo][1]
        self.vel = 0
        self.direction = vec2(0,1)
        self.direction = self.direction.rotate(startingPositinos[trackNo][2])
        self.acc = 0
        self.width = 40
        self.height = 20
        self.turningRate = 5.0 / self.width
        self.friction = 0.98
        self.maxSpeed = self.width / 4.0
        self.maxReverseSpeed = -1 #self.maxSpeed / 16.0 is used as a minimum for speed
        self.accelerationSpeed = self.width / 160.0
        self.dead = False
        self.driftMomentum = 0
        self.driftFriction = 0.87
        self.lineCollisionPoints = []
        self.collisionLineDistances = []
        self.vectorLength = 1500

        self.carPic = pyglet.image.load('./Data/car2.png')
        self.carSprite = pyglet.sprite.Sprite(self.carPic, x=self.x, y=self.y)
        self.carSprite.update(rotation=0, scale_x=self.width / self.carSprite.width,
                              scale_y=self.height / self.carSprite.height)

        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False
        self.walls = walls
        self.rewardGates = rewardGates
        self.rewardNo = 0

        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - vec2(self.x, self.y)

        self.reward = 0

        self.score = 0
        self.lifespan = 0
    """
    draws the car to the screen
    """

    def reset(self):
        global vec2

        self.timeAttempt=time.time()
        self.timeLap=time.time()

        self.x = startingPositinos[trackNo][0]
        self.y = startingPositinos[trackNo][1]
        self.vel = 0
        self.direction = vec2(0,1)
        self.direction = self.direction.rotate(startingPositinos[trackNo][2])
        self.acc = 0
        self.dead = False
        self.driftMomentum = 0
        self.lineCollisionPoints = []
        self.collisionLineDistances = []

        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False
        self.rewardNo = 0
        self.reward = 0

        self.lifespan = 0
        self.score = 0
        for g in self.rewardGates:
            g.active = True

    def show(self):
        # first calculate the center of the car in order to allow the
        # rotation of the car to be anchored around the center
        upVector = self.direction.rotate(90)
        drawX = self.direction.x * self.width / 2 + upVector.x * self.height / 2
        drawY = self.direction.y * self.width / 2 + upVector.y * self.height / 2
        self.carSprite.update(x=self.x - drawX, y=self.y - drawY, rotation=-get_angle(self.direction))
        self.carSprite.draw()
        # self.showCollisionVectors()

    """
     returns a vector of where a point on the car is after rotation 
     takes the position desired relative to the center of the car when the car is facing to the right
    """

    def getPositionOnCarRelativeToCenter(self, right, up):
        global vec2
        w = self.width
        h = self.height
        rightVector = vec2(self.direction)
        rightVector.normalize()
        upVector = self.direction.rotate(90)
        upVector.normalize()

        return vec2(self.x, self.y) + ((rightVector * right) + (upVector * up))

    def updateWithAction(self, actionNo):
        self.turningLeft = False
        self.turningRight = False
        self.accelerating = False
        self.reversing = False

        if actionNo == 0:
            self.turningLeft = True
        elif actionNo == 1:
            self.turningRight = True
        elif actionNo == 2:
            self.accelerating = True
        elif actionNo == 3:
            self.reversing = True
        elif actionNo == 4:
            self.accelerating = True
            self.turningLeft = True
        elif actionNo == 5:
            self.accelerating = True
            self.turningRight = True
        elif actionNo == 6:
            self.reversing = True
            self.turningLeft = True
        elif actionNo == 7:
            self.reversing = True
            self.turningRight = True
        elif actionNo == 8:
            pass
        totalReward = 0

        for i in range(1):
            if not self.dead:
                self.lifespan+=1

                self.updateControls()
                self.move()

                if self.hitAWall():
                    #self.dead = True
                    self.reset()
                    # return
                self.checkRewardGates()
                totalReward += self.reward

        self.setVisionVectors()

        # self.update()

        self.reward = totalReward

    """
    called every frame
    """

    def update(self):
         
        if not self.dead:
            self.updateControls()
            self.move()

            if self.hitAWall():
                #self.dead = True
                self.reset()
                # return
            self.checkRewardGates()
            self.setVisionVectors()

    def checkRewardGates(self):
        global vec2
        self.reward = -1
        if self.rewardGates[self.rewardNo].hitCar(self):
            self.rewardGates[self.rewardNo].active = False
            self.rewardNo += 1
            self.score += 1
            self.reward = 10
            if self.rewardNo == len(self.rewardGates):
                self.rewardNo = 0
                for g in self.rewardGates:
                    g.active = True
                
                if(time.time()-self.timeLap<self.bestLapTime):
                    self.bestLapTime=time.time()-self.timeLap
                self.timeLap=time.time()
        self.directionToRewardGate = self.rewardGates[self.rewardNo].center - vec2(self.x, self.y)

    """
    changes the position of the car to account for acceleration, velocity, friction and drift
    """

    def move(self):
        global vec2
        self.vel += self.acc
        self.vel *= self.friction
        self.constrainVel()

        driftVector = vec2(self.direction)
        driftVector = driftVector.rotate(90)

        addVector = vec2(0, 0)
        addVector.x += self.vel * self.direction.x
        addVector.x += self.driftMomentum * driftVector.x
        addVector.y += self.vel * self.direction.y
        addVector.y += self.driftMomentum * driftVector.y
        self.driftMomentum *= self.driftFriction

        if addVector.length() != 0:
            addVector.normalize()

        addVector.x * abs(self.vel)
        addVector.y * abs(self.vel)

        self.x += addVector.x
        self.y += addVector.y

    """
    keeps the velocity of the car within the maximum and minimum speeds
    """

    def constrainVel(self):
        if self.maxSpeed < self.vel:
            self.vel = self.maxSpeed
        elif self.vel < self.maxReverseSpeed:
            self.vel = self.maxReverseSpeed

    """
    changes the cars direction and acceleration based on the users inputs
    """

    def updateControls(self):
        multiplier = 1
        if abs(self.vel) < 5:
            multiplier = abs(self.vel) / 5
        if self.vel < 0:
            multiplier *= -1

        driftAmount = self.vel * self.turningRate * self.width / (9.0 * 8.0)
        if self.vel < 5:
            driftAmount = 0

        if self.turningLeft:
            self.direction = self.direction.rotate(radiansToAngle(self.turningRate) * multiplier)

            self.driftMomentum -= driftAmount
        elif self.turningRight:
            self.direction = self.direction.rotate(-radiansToAngle(self.turningRate) * multiplier)
            self.driftMomentum += driftAmount
        self.acc = 0
        if self.accelerating:
            if self.vel < 0:
                self.acc = 3 * self.accelerationSpeed
            else:
                self.acc = self.accelerationSpeed
        elif self.reversing:
            if self.vel > 0:
                self.acc = -2 * self.accelerationSpeed
            else:
                self.acc = 0
                self.vel = 0

    """
    checks every wall and if the car has hit a wall returns true    
    """

    def hitAWall(self):
        for wall in self.walls:
            if wall.hitCar(self):
                #print(self.x,self.y)
                return True

        return False

    """
    returns the point of collision of a line (x1,y1,x2,y2) with the walls, 
    if multiple walls are hit it returns the closest collision point
    """

    def getCollisionPointOfClosestWall(self, x1, y1, x2, y2):
        global vec2
        minDist = 2 * displayWidth
        closestCollisionPoint = vec2(0, 0)
        for wall in self.walls:
            collisionPoint = getCollisionPoint(x1, y1, x2, y2, wall.x1, wall.y1, wall.x2, wall.y2)
            if collisionPoint is None:
                continue
            if dist(x1, y1, collisionPoint.x, collisionPoint.y) < minDist:
                minDist = dist(x1, y1, collisionPoint.x, collisionPoint.y)
                closestCollisionPoint = vec2(collisionPoint)
        return closestCollisionPoint

    """
    by creating lines in many directions from the car and getting the closest collision point of that line
    we create  "vision vectors" which will allow the car to 'see' 
    kinda like a sonar system
    """

    def getState(self):
        self.setVisionVectors()
        normalizedVisionVectors = [1 - (max(1.0, line) / self.vectorLength) for line in self.collisionLineDistances]

        normalizedForwardVelocity = max(0.0, self.vel / self.maxSpeed)
        normalizedReverseVelocity = max(0.0, self.vel / self.maxReverseSpeed) #do wyjebania, chyba że przywrócimy wsteczyn ale raczej nah xD
        if self.driftMomentum > 0:
            normalizedPosDrift = self.driftMomentum / 5
            normalizedNegDrift = 0
        else:
            normalizedPosDrift = 0
            normalizedNegDrift = self.driftMomentum / -5

        normalizedAngleOfNextGate = (get_angle(self.direction) - get_angle(self.directionToRewardGate)) % 360
        if normalizedAngleOfNextGate > 180:
            normalizedAngleOfNextGate = -1 * (360 - normalizedAngleOfNextGate)

        normalizedAngleOfNextGate /= 180

        normalizedState = [*normalizedVisionVectors, normalizedForwardVelocity, normalizedReverseVelocity,
                           normalizedPosDrift, normalizedNegDrift, normalizedAngleOfNextGate]
        return np.array(normalizedState)

    def setVisionVectors(self):
        self.collisionLineDistances = []
        self.lineCollisionPoints = []
        for i in self.angles :
            self.setVisionVector(0,0,i)

    """
    calculates and stores the distance to the nearest wall given a vector 
    """

    def setVisionVector(self, startX, startY, angle):
        collisionVectorDirection = self.direction.rotate(angle)
        collisionVectorDirection = collisionVectorDirection.normalize() * self.vectorLength
        startingPoint = self.getPositionOnCarRelativeToCenter(startX, startY)
        collisionPoint = self.getCollisionPointOfClosestWall(startingPoint.x, startingPoint.y,
                                                             startingPoint.x + collisionVectorDirection.x,
                                                             startingPoint.y + collisionVectorDirection.y)
        if collisionPoint.x == 0 and collisionPoint.y == 0:
            self.collisionLineDistances.append(self.vectorLength)
        else:
            self.collisionLineDistances.append(
                dist(startingPoint.x, startingPoint.y, collisionPoint.x, collisionPoint.y))
        self.lineCollisionPoints.append(collisionPoint)

    """
    shows dots where the collision vectors detect a wall 
    """

    def showCollisionVectors(self):
        global drawer
        for point in self.lineCollisionPoints:
            drawer.setColor([255, 0, 0])
            drawer.circle(point.x, point.y, 5)