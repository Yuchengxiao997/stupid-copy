import pybullet as p
import pybullet_data as pd
import math

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
print(pd.getDataPath())
pandaUid=p.loadURDF("kuka_iiwa/model.urdf",useFixedBase=True)
tableUid=p.loadURDF("table/table.urdf",basePosition=[0.5,0,-0.65])
bp = [0.1,0.1,0.4]
object_id = p.loadURDF("random_urdfs/002/002.urdf",basePosition=bp,useFixedBase=True)
#dog = p.loadURDF("husky/husky.urdf",basePosition=bp,useFixedBase=True)
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
print("碰撞检测：")
print(p.getCollisionShapeData(pandaUid,2))


while True:
    p.stepSimulation()