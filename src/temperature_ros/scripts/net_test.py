#!/usr/bin/env python3
import numpy as np
import sys

sys.path.append('/home/leo/Desktop/temperature_ws/src/')

from temperature_ros.modules.temperature_ros.MaterialClassifier import MaterialDataset 
from temperature_ros.modules.temperature_ros.MaterialClassifier import MaterialClassifier 

from temperature_ros.modules.temperature_ros.MaterialClassifier import LSTMNet 
import torch

# =============================================================================
# Main
# =============================================================================  
def main():
    dataset = MaterialDataset('/home/leo/Desktop/temperature_ws/src/temperature_ros/config/net/data')
    net = MaterialClassifier(dataset)


    net.load('material_classifier_model_C5')

    #torch.save(net.net.state_dict(), '/home/leo/Desktop/temperature_ws/src/temperature_ros/config/net/weights/state_dict.pt')
    torch.export.save(net.net, '/home/leo/Desktop/temperature_ws/src/temperature_ros/config/net/weights/state_dict.pt2')
    
    x = np.ones((381,1))
    t = np.arange(0,40,40/381).reshape((381,1))

    (score, pred) = net.predict(np.hstack([x,t]))
    print(score)
    print(pred)
    pass
        

if __name__ == "__main__":
    main()
