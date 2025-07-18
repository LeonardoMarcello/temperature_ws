Per abilitare le letture tramite seriale è necessario eseguire il seguente comando
> sudo chmod 666 /dev/ttyACM0
> sudo chmod 666 /dev/ttyUSB0

Per cambiare la funzione caratteristica del sensore di temperatura modificare il file temperature_utils.h a riga 8 e rifare catkin_make

Per selezionare l'intorno di ciascun ogetto da graspare (per la bounding box) modificare bounding_box_node.py a riga 72



ALTRO:
Esempio d'uso per fare il broadcast di una poincloud già salvata
> rosrun pcl_ros pcd_to_pointcloud src/temperature_ros/pcd/drill0.pcd 0.1

Per Testare il controllo del franka si può usare il nodo
> rosrun temperature_ros robot_controller.py



SOFTHAND:
control launch
> roslaunch qb_hand_control control_qbhand.launch standalone:=true activate_on_initialization:=true use_controller_gui:=false

topic closure (cli: tab tab per inserire messaggio default):
> rostopic pub /qbhand1/control/qbhand1_synergy_trajectory_controller/command trajectory_msgs/JointTrajectory



ESPERIMENTI:
Il launch per il framework del setup sperimentale: 
> roslaunch temperature_ros demo.launch

Il nodo per eseguire gli esperimenti: 
> rosrun temperature_ros experiment.py

Per attivare il passaggio di stato usare il comando:
> rostopic pub /activate std_msgs/Empty "{}"

Salva dati
> rosbag record -O workdir/data_2105251030.bag /rosout /tf /temperature /yolov7/visualization /camera/color/image_raw


TO DO: 
--> Calibrazione (fare .csv per le due situazioni):
    --> CALIBRAZIONE A TEMPERATURE (loro con Tamb, 35°C, 50°C) [✔️]
    --> CALIBRAZIONE A MATERIALI (loro a 7°C con: Metallo, Silicone, Legno, Plastica) [✔️]
--> Finire Setup: 
    --> Montaggio scheda lettura [✔️]
    --> Rafforzare cablaggio fingertip [✔️]
    --> Test di presa [✔️] 
    --> ristampare case circuito [✔️]
    --> (Opz.) Sistemare acquisizioni RealSense ROS [✔️]
    --> (Opz.) Sistemare comunicazione per integrare visione su 1 PC [✔️]



L-P-A x 2 - material_260620251535.bag - 40 °C
P-L-A x 2 - material_260620251651.bag - 40 °C