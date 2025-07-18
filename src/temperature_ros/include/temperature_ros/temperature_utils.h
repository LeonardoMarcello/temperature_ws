#ifndef TEMPERATURE_UTILS_H_
#define TEMPERATURE_UTILS_H_

static inline double exponentialSmoothing(double current_raw_value, double last_smoothed_value, double alpha){
    return alpha*current_raw_value + (1-alpha)*last_smoothed_value;
}
static inline double adc2temperature(int adc_raw_value){
    // linear model : V = 1000.0*(0.01029 T + 2.0465)
    //double temperature = (adc_raw_value/1000.0 - 2.0465)/0.01029;
    //double temperature = -0.02*adc_raw_value + 1.72;
    double temperature = -45.61*adc_raw_value + 76.7;
    
    return temperature;
}

static inline double adc2temperature(double adc_raw_value, double adc_bias){
    // linear model : V = 1000.0*(0.01029 T + 2.0465)
    //double temperature = (adc_raw_value/1000.0 - 2.0465)/0.01029;
    //double temperature = -0.02*adc_raw_value + 1.72;
    double temperature = -45.61*(adc_raw_value-adc_bias) + 76.7;
    //double temperature = 1.0*adc_raw_value-adc_bias;
    
    return temperature;
}

bool compute_bias(double &bias, double V, int count, int num_of_point){
    // update runnig bias
    if (count == 0){bias = 0;}
    bias += V;

    // when all sample are collected compute mean and return true otherwise return false
    if (count < num_of_point){
        return false;
    }
    else{
        bias = bias/(num_of_point+1)  - 1.15; // Set bias to obtain 1.15V a Tamb.
        return true;
    }
}

#endif // TEMPERATURE_UTILS_H_