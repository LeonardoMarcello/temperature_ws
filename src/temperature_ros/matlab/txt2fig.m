clear all
close all

% Set default LaTeX formatting and font sizes for all plots
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultAxesFontSize', 15);
set(groot, 'defaultTextFontSize', 15);


%% Load txt
addpath(".\25_06_2025")
addpath(".\26_06_2025")
Files = [dir(".\26_06_2025\*.txt")];%,dir(".\26_06_2025\*.csv")];

use_window = false;
%use_window = true;
time_window = [530, 1060]; % intervallo in [s]

% addpath("E:\LEO\- Universita leo\Phd\sensori milano\CIMaINa\caratterizzazione\")
% csvFiles = dir("E:\LEO\- Universita leo\Phd\sensori milano\CIMaINa\caratterizzazione\*.txt");

% addpath("C:\Users\leona\Desktop\sensori milano\experiment\")
% csvFiles = dir("C:\Users\leona\Desktop\sensori milano\experiment\*.txt");

fileNames = {Files.name};  % cell array of file names
num_of_file = size(fileNames,2);

T = dictionary; 

for file_idx = 1:num_of_file
    filename = fileNames{file_idx};
    T{filename} = readtable(filename);
end


%% signal
for file_idx = 1:num_of_file
    filename = fileNames{file_idx};
    T{filename} = readtable(filename);
    if filename == "note.txt"
        continue
    end

    t = T{filename};
    try
        % ts = timeseries(t.x_temperature_temperature, (t.x__temperature_recceipt_time-t.x__temperature_recceipt_time(1)));
        ts = timeseries(t.x_delta_temperature_temperature(1:end-15), (t.x_delta_temperature_recceipt_time(1:end-15)-t.x_delta_temperature_recceipt_time(1)));
    
    catch
        ts = timeseries(t.ch5, (t.time-t.time(1)));
    end
    
    if use_window
        ts = getsampleusingtime(ts,time_window(1), time_window(2));
    end
    
    [unique_time, idx] = unique(ts.Time);
    unique_data = ts.Data(idx);

    
    figure
    hold on
    plot(ts,'DisplayName','signal')
    
    grid on
    legend
    ylabel('Voltage [V]')
    xlabel('Time [s]')
    xtickangle(45)
    xticks(0:60:unique_time(end))
    title(filename)
end



%% apply conversion
for file_idx = 1:num_of_file
    filename = fileNames{file_idx};
    T{filename} = readtable(filename);
    if filename == "note.txt"
        continue
    end

    t = T{filename};
    try
        ts = timeseries(-45.61*t.x_temperature_temperature+76.70, (t.x__temperature_recceipt_time-t.x__temperature_recceipt_time(1)));
    catch
        ts = timeseries(-45.61*t.ch5+76.70, (t.time-t.time(1)));
    end
    
    if use_window
        ts = getsampleusingtime(ts,time_window(1), time_window(2));
    end


    ts_sensor = ts;

    [unique_time, idx] = unique(ts.Time);
    unique_data = ts.Data(idx);


    figure
    hold on
    plot(ts,'DisplayName','signal')

    grid on
    legend
    ylabel('Temperature [°C]')
    xlabel('Time [s]')
    xtickangle(45)
    xticks(0:240:unique_time(end))
    title(filename)
end


%% FLIR
% %% Load txt
% Files = dir(".\25_06_2025\*.csv");
% 
% use_window = false;
% %use_window = true;
% time_window = [530, 1060]; % intervallo in [s]
% 
% % addpath("E:\LEO\- Universita leo\Phd\sensori milano\CIMaINa\caratterizzazione\")
% % csvFiles = dir("E:\LEO\- Universita leo\Phd\sensori milano\CIMaINa\caratterizzazione\*.txt");
% 
% % addpath("C:\Users\leona\Desktop\sensori milano\experiment\")
% % csvFiles = dir("C:\Users\leona\Desktop\sensori milano\experiment\*.txt");
% 
% fileNames = {Files.name};  % cell array of file names
% num_of_file = size(fileNames,2);
% 
% T = dictionary; 
% 
% for file_idx = 1:num_of_file
%     filename = fileNames{file_idx};
%     T{filename} = readtable(filename);
% end
% 
% 
% %% signal
% for file_idx = 1:num_of_file
%     filename = fileNames{file_idx};
%     T{filename} = readtable(filename);
%     t = T{filename};
%     ts = timeseries(t.Ellipse1_C_, t.reltime);
% 
%     ts_flir = ts;
% 
%     if use_window
%         ts = getsampleusingtime(ts,time_window(1), time_window(2));
%     end
% 
%     [unique_time, idx] = unique(ts.Time);
%     unique_data = ts.Data(idx);
% 
% 
%     figure
%     hold on
%     plot(ts,'DisplayName','signal')
% 
%     grid on
%     legend
%     ylabel('Voltage [V]')
%     xlabel('Time [s]')
%     xtickangle(45)
%     xticks(0:60:unique_time(end))
%     title(filename)
% end


%% COMBO
% ts_sensor = getsampleusingtime(ts_sensor,17,ts_sensor.Time(end)); 
% ts_sensor.Time = ts_sensor.Time - ts_sensor.Time(1);
% 
% ts_flir = getsampleusingtime(ts_flir,277,ts_flir.Time(end)); 
% ts_flir.Time = ts_flir.Time - ts_flir.Time(1);
% 
% 
% figure
% hold on
% plot(ts_sensor,'DisplayName','signal')
% plot(ts_flir,'DisplayName','flir')
% 
% grid on
% legend
% ylabel('Temperature [°C]')
% xlabel('Time [s]')
% xtickangle(45)
% xticks(0:240:unique_time(end))
% yticks(15:1:45)
% title(filename)