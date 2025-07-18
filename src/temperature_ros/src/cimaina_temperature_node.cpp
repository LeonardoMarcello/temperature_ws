#include "ros/ros.h"
#include "std_msgs/Empty.h"
#include "sensor_msgs/Temperature.h"
#include "temperature_ros/temperature_utils.h"
#include "temperature_ros/fading_filter.hpp"
#include <Eigen/Dense>
#include <Eigen/LU>

#include <libserial/SerialPort.h>



using namespace std;

// Bias callback
bool running_bias = false;
void biasCallback(const std_msgs::Empty::ConstPtr& msg){
  running_bias = true;
}

// Main
int main(int argc, char **argv){
  ros::init(argc, argv, "temperature_driver");
  ros::NodeHandle nh;

  ROS_INFO("Hi from CIMaINa temperature_node");

  // Variables
  double rate, alpha, beta, last_smoothed_value;
  last_smoothed_value = -1;

  // communications settings
  LibSerial::SerialPort serial;
  std::string serial_port_name; 

  double adc[3];
  double adc_bias = 0.0;                // measurements bias
  int count = 0; int num_of_point = 10; // num of samples for bias computation

  // parameters
  nh.param<std::string>("/serial_port_name", serial_port_name, "/dev/ttyACM0");  // USB device name
  nh.param<double>("/cimaina_temperature_node/rate", rate, 10);                                           // reader rate [Hz]
  nh.param<double>("/cimaina_temperature_node/alpha", alpha, 1);                                          // exponential filter params: x = ay-(1-a)x
                                                                                                          // (alpha in [0,1]) 
  nh.param<double>("/cimaina_temperature_node/beta", beta, .7);                                          // fading filter params: (beta in [0,1])
  

  // variable
  fading_filter::FadingFilter temperature_ff = fading_filter::FadingFilter("temperature_filter", beta, 1, 0);


  // init communications
  try{        
    serial.Open(serial_port_name);
    serial.SetBaudRate(LibSerial::BaudRate::BAUD_9600);
  } catch (const LibSerial::OpenFailed& e) {
    ROS_ERROR("Failed to open serial port %s: %s", serial_port_name.c_str(), e.what());
    return -1;
  } catch (const std::exception& e) {
    ROS_ERROR("Error %s",e.what());
    return -1;
  }

  // publisher
  ros::Publisher temperature_pub = nh.advertise<sensor_msgs::Temperature>("temperature", 100);
  ros::Publisher delta_temperature_pub = nh.advertise<sensor_msgs::Temperature>("delta_temperature", 100);
  // Subscriber
  ros::Subscriber bias_sub = nh.subscribe("/set_bias", 100, biasCallback);

  // Set node rate
  ros::Rate ros_rate(rate);
  while (ros::ok()) {
    // read raw voltage value
    if (serial.IsDataAvailable()) {
      std::string data;
      serial.ReadLine(data, '\n', 100);
      adc[0] = std::stof(data);

      // filter the signal
      /*  The definition of the exponential smoothing and the adc2temperature function
          can be found in the temperature_utils.h file for the sake of readability */
      double filtered_adc;
      if (last_smoothed_value != -1) {
        filtered_adc = exponentialSmoothing(adc[0], last_smoothed_value, alpha);
      }
      else {
        filtered_adc = adc[0];
      }
      double dvdt = (last_smoothed_value-filtered_adc)*rate;
      last_smoothed_value = filtered_adc;

      // whether compute bias
      if (running_bias){
        bool res = compute_bias(adc_bias, last_smoothed_value, count, num_of_point);
        count += 1;
        if(res){
          ROS_INFO("Bias computed on %d Samples. Signal at rest = %.2f", num_of_point, adc_bias);
          running_bias = !running_bias;
          count = 0;
        }
      }

      // convert ADC raw value to temperature
      //double temperature = adc2temperature(filtered_adc, adc_bias); // convert into temperature
      double temperature = filtered_adc - adc_bias; // Shift temperature at rest in 1V
      temperature_ff.update(temperature, 1/rate);

      // Publish message
      sensor_msgs::Temperature msg;
      msg.header.stamp = ros::Time::now();
      msg.header.frame_id = "temperature";
      msg.temperature = temperature;
      msg.variance = 0;
      temperature_pub.publish(msg); 


      // Publish message
      sensor_msgs::Temperature delta_msg;
      delta_msg.header.stamp = ros::Time::now();
      delta_msg.header.frame_id = "delta_temperature";
      delta_msg.temperature = temperature_ff.dx;
      delta_msg.variance = 0;
      delta_temperature_pub.publish(delta_msg); 
    }
    else{
        //ROS_WARN("Couldn't retrieve measurements.");
    }

    ros::spinOnce();
    ros_rate.sleep();

  }

  serial.Close();
  return 0;
}