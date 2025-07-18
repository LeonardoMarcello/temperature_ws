#include "ros/ros.h"
#include "sensor_msgs/Temperature.h"
#include "temperature_ros/qbmove_communications.h"
#include "temperature_ros/cp_communications.h"
#include "temperature_ros/temperature_utils.h"
#include <Eigen/Dense>
#include <Eigen/LU>


using namespace std;


// Main
int main(int argc, char **argv){
  ros::init(argc, argv, "temperature_driver");
  ros::NodeHandle nh;

  ROS_INFO("Hi from temperature_node");

  // Variables
  double rate, alpha, last_smoothed_value;
  last_smoothed_value = -1;

  // communications settings
  comm_settings   comm_settings_t;
  std::string serial_port_name; 
  int device_id;

  int num_of_channels = 1;
  short int adc[3];

  // parameters
  nh.param<std::string>("/serial_port_name", serial_port_name, "/dev/ttyUSB0");
  nh.param<int>("/device_id", device_id, 1);  
  nh.param<double>("/rate", rate, 50);
  nh.param<double>("/alpha", alpha, 1);
  

  // init communications
  openRS485(&comm_settings_t, serial_port_name.c_str());
  if(comm_settings_t.file_handle == INVALID_HANDLE_VALUE)
  {    
    ROS_WARN("HANDLE: %d", comm_settings_t.file_handle);
    // check for connection
    int     i, num_ports;
    char    list_of_ports[10][255];

    num_ports = RS485listPorts(list_of_ports);
    ROS_WARN("Detected Ports:");    
    for(i = 0; i < num_ports; ++i){
      ROS_WARN(">> Port %d: %s", i, list_of_ports[i]);
    }
    return -1;
  }

  // publisher
  ros::Publisher temperature_pub = nh.advertise<sensor_msgs::Temperature>("temperature", 100);

  // Set node rate
  ros::Rate ros_rate(rate);
  while (ros::ok()) {
    
    // read ADC Raw
    if(!commGetADCRawValues(&comm_settings_t, device_id, num_of_channels, adc)){
        //printf("ADC raw: %d\n",adc[0]);
        
        // filter the signal
        /* 
          The definition of the exponential smoothing and the adc2temperature function
           can be found in the temperature_utils.h file for the sake of readability
        */
        double filtered_adc;
        if (last_smoothed_value != -1) {
        filtered_adc = exponentialSmoothing(adc[0], last_smoothed_value, alpha);
        }
        else {
          filtered_adc = adc[0];
        }
        last_smoothed_value = filtered_adc;

        // convert ADC raw value to temperature
        double temperature = adc2temperature(filtered_adc);

        // Publish message
        sensor_msgs::Temperature msg;
        msg.header.stamp = ros::Time::now();
        msg.header.frame_id = "temperature";
        msg.temperature = temperature;
        temperature_pub.publish(msg); 
    }
    else{
        ROS_WARN("Couldn't retrieve measurements.");
    }

    ros::spinOnce();
    ros_rate.sleep();

  }

  closeRS485(&comm_settings_t);
  return 0;
}