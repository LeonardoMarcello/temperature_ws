clear all
close all

% Set default LaTeX formatting and font sizes for all plots
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultAxesFontSize', 15);
set(groot, 'defaultTextFontSize', 15);


%% raw plot
data = [1.25,1.247,1.259,0.752,0.777,0.799,1.222,1.084,0.87,1.067,1.198,0.927,1.179,1.091,0.96,1.07,1.082,1.088,0.966,1.171;
        18.4,18.9,19.1,41.1,40.1,39.5,20.9,27.3,36.8,27.6,21.8,35.3,22.6,27.7,35.2,27.8,27.8,27.7,34.3,23.5];
figure
hold on
plot(data(1,:), data(2,:),'o','DisplayName','signal')

grid on
legend
xlabel('Voltage [V]')
ylabel('Temperature [°C]')
xtickangle(45)
% xticks(0:60:unique_time(end))
title("Experiment")

%% fit
[c_fit,info] = fit(data(1,:)',data(2,:)','a*x+b');
%[c_fit,info] = fit(data(1,:)',data(2,:)','-39.8406*x+b');

figure
hold on
plot(1:20, data(2,:),'o','DisplayName','$T_{real}$')
plot(1:20, c_fit(data(1,:)),'x','DisplayName','$\hat{T}$')
yregion(25, 30,'FaceColor','y','EdgeColor','k','DisplayName','Warm')
yregion(15, 25,'FaceColor','g','EdgeColor','k','DisplayName','Cold')
yregion(30, 45,'FaceColor','r','EdgeColor','k','DisplayName','Hot')

grid on
legend
xlabel('Trial')
ylabel('Temperature [°C]')
xtickangle(45)
xticks(1:21)
title("Experiment")






tmp = -0.0251\(data(1,:) - 1.8329)-2;
figure
hold on
plot(data(1,:), data(2,:),'o','DisplayName','groundtruth')
% plot(data(1,:), tmp, 'x', 'DisplayName','hat')
plot(data(1,:), c_fit(data(1,:)), 'x', 'DisplayName','hat')
yregion(25, 30,'FaceColor','y','EdgeColor','k','DisplayName','Warm')
yregion(15, 25,'FaceColor','g','EdgeColor','k','DisplayName','Cold')
yregion(30, 45,'FaceColor','r','EdgeColor','k','DisplayName','Hot')

grid on
legend
xlabel('Voltage [V]')
ylabel('Temperature [°C]')
xtickangle(45)
% xticks(0:60:unique_time(end))
title("Experiment")


errors = abs(data(2,:)' - c_fit(data(1,:)));
mean(errors)
std(errors)
rmse(c_fit(data(1,:)), data(2,:)')

figure
hold on
bar(errors,'DisplayName','$e_T$')
grid on
legend
xlabel('Trial')
ylabel('Temperature Error [°C]')
xtickangle(45)
xticks(1:21)
title("Experiment")