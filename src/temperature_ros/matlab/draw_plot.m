clear all
close all

% Set default LaTeX formatting and font sizes for all plots
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultAxesFontSize', 15);
set(groot, 'defaultTextFontSize', 15);


%% Load Experiment txt
colors = {[0.12156863 0.46666667 0.70588235]
 [0.83921569 0.15294118 0.15686275]
 [0.89019608 0.46666667 0.76078431]
 [0.09019608 0.74509804 0.81176471]};

names = {'aluminium','wood','plastic','glass'};
% colors = [
%     0.000, 0.447, 0.741;  % blue
%     0.850, 0.325, 0.098;  % orange
%     0.929, 0.694, 0.125;  % yellow
%     0.494, 0.184, 0.556;  % purple
%     0.466, 0.674, 0.188;  % green
%     0.301, 0.745, 0.933;  % cyan
%     0.635, 0.078, 0.184;  % dark red
%     0.333, 0.333, 0.333;  % dark gray
%     0.600, 0.600, 0.600;  % light gray
%     0.098, 0.325, 0.850;  % deep blue
% ];
% 
% voltage = [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9];
% dvoltage = [1,2,3,4,5,6,7,8,9,10];
% times = [1,2,3,4,5,6,7,8,9,10];
addpath(".\csv_to_plot")
data = readtable('csv_to_plot_a.csv'); % to_skip = 5
data = readtable('csv_to_plot_l.csv'); % to_skip = 1
% data = readtable('csv_to_plot_p.csv'); % to_skip = 1
% data = readtable('csv_to_plot_v.csv'); % to_skip = 1

% Access columns by name
voltage = data.Temperature;
dvoltage = data.dTemperature;
times = 1:402;


% 
% NAME = 'alluminio';         % <---------------------------- Set classification results
% CLASS = 1;
% SCORE = [9.99979615e-01 4.67421515e-08 1.47840808e-06 1.89486545e-05];

NAME = 'legno';
CLASS = 2;
SCORE = [2.05227971e-05 9.12569523e-01 8.05648938e-02 6.84503745e-03];

% NAME = 'plastica';
% CLASS = 3;
% SCORE = [4.23189013e-05 1.72047839e-05 6.32538021e-01 3.67402524e-01];

% NAME = 'vetro';
% CLASS = 4;
% SCORE = [2.46076888e-05 1.17030786e-05 4.89474684e-01 5.10488987e-01];

NAME = names{CLASS};
SAVE = 1;

%% Plot =========================================================
if SAVE == 1
    myVideo = VideoWriter(NAME,"MPEG-4");
    myVideo.FrameRate = 10;
    open(myVideo);
end

figure(1)
for i=1:length(voltage)
    if i<length(voltage)-1
        color='black';
    else
        color=colors{CLASS};
    end

    % plot voltage -----------------------
    subplot(2,2,1)
    plot(times(i)/10, voltage(i),'s','MarkerFaceColor',color)
    plot(times(1:i)/10, voltage(1:i),'-','Color',color)
    axis([0 40.3 0.7 1.2])
    % text(EXP{i,2}+0.1, EXP{i,1}, "(" + EXP{i,2} + ", " + EXP{i,1} + ")")
    % title('Values')
    ylabel 'Signal [V]'
    xlabel 'Time [s]'
    grid on
        
    % plot dvoltage -----------------------
    subplot(2,2,2)
    plot(times(i)/10, dvoltage(i),'s','MarkerFaceColor', color)
    plot(times(1:i)/10, dvoltage(1:i),'-','Color', color)
    axis([0 40.3 -0.15 0.01])
    % text(EXP{i,2}+0.1, EXP{i,1}, "(" + EXP{i,2} + ", " + EXP{i,1} + ")")
    % title('Values')
    ylabel 'Derivative [V/s]'
    xlabel 'Time [s]'
    grid on




    % plot guess -----------------------
    subplot(2,2,3)
    set(gca,'visible','off')
    if i<length(voltage)-1
        b = bar([0, 0, 0, 0]);
    else
        % Define the 4 values
        values = SCORE;
        
        % Create the bar plot
        b = bar(values);
    end
    
    for j = 1:length(SCORE)
        b.FaceColor = 'flat'; % Enable per-bar coloring
        b.CData(j, :) = colors{j};
    end
        
    % Optional: Add labels and title
    ylim([0 1])
    xticklabels(names); % Custom x-axis labels
    xtickangle(45)
    title('Prediction');
    grid on
        
    % plot guess -----------------------
    subplot(2,2,4)
    set(gca,'visible','off')
    cla
    if i<length(voltage)-1
        % load_txt = repelem([' ','.',' '], [mod(i,3),1,2-mod(i,3)]);
        load_txt = '';
    else
        load_txt = NAME;
    end
    t = text(0.5,0.5,load_txt,"FontName",'Cambria','FontSize',30, Color=color);
    set(t,'visible','on','HorizontalAlignment','center','VerticalAlignment','middle')
    
    pause(1/10)
    if SAVE == 1
        frame = getframe(gcf);
        writeVideo(myVideo, frame);
    end
end

if SAVE == 1
    close(myVideo);
    fprintf("Video saved as "+NAME+".mp4\n");
end
