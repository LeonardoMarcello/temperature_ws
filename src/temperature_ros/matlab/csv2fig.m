clear all
close all

% Set default LaTeX formatting and font sizes for all plots
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultAxesFontSize', 15);
set(groot, 'defaultTextFontSize', 15);


%% Load txt
addpath(".\08_07_2025")
addpath(".\14_07_2025")
addpath(".\train")
addpath(".\test")
addpath(".\full")
% Files = [dir(".\08_07_2025\*.csv"); dir(".\14_07_2025\*.csv")];
% Files = [dir(".\test\*.csv")];
Files = [dir(".\full\*.csv")];

colors = [
    0.000, 0.447, 0.741;  % blue
    0.850, 0.325, 0.098;  % orange
    0.929, 0.694, 0.125;  % yellow
    0.494, 0.184, 0.556;  % purple
    0.466, 0.674, 0.188;  % green
    0.301, 0.745, 0.933;  % cyan
    0.635, 0.078, 0.184;  % dark red
    0.333, 0.333, 0.333;  % dark gray
    0.600, 0.600, 0.600;  % light gray
    0.098, 0.325, 0.850;  % deep blue
];

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

    t = T{filename};
       
    experiments = unique(t.Experiment);




    for i = 1:length(experiments)
        exp_id = experiments(i);
        
        % Select rows for this experiment
        exp = t.Experiment == exp_id;
        time = 1e-9*t.Timestamp(exp);
        time = time - time(1);
        temperature = t.temperature_C_(exp);
        dtemperature = t.delta_t_C_s_(exp);
        notes = t.notes(exp);
        if contains(notes,"alluminio")
            color_idx = 1;
        elseif contains(notes,"plastica")
            color_idx = 2;
        elseif contains(notes,"vetro")
            color_idx = 3;
        elseif contains(notes,"legno")
            color_idx = 4;
        end          

        T_ts = timeseries(temperature, time);
        %T_ts = timeseries(temperature-temperature(1)+1, time);
        dT_ts = timeseries(dtemperature, time);
        
        % Plot
        figure(1)
        hold on
        if exp_id==1
            plot(T_ts,'.','Color', colors(color_idx,:),'HandleVisibility', 'on','DisplayName', notes{1})
        else
            plot(T_ts,'.','Color', colors(color_idx,:),'HandleVisibility', 'off')
        end
        grid on
        legend
        ylabel('Voltage [V]')
        xlabel('Time [s]')
        xtickangle(45)
        % xticks(0:60:unique_time(end))
        title('Temperature [°C]')

        figure(2)
        hold on        
        if exp_id==1
            plot(dT_ts,'.','Color', colors(color_idx,:),'HandleVisibility', 'on','DisplayName', notes{1})
        else
            plot(dT_ts,'.','Color', colors(color_idx,:),'HandleVisibility', 'off')
        end
        
        grid on
        legend
        ylabel('Voltage [V]')
        xlabel('Time [s]')
        xtickangle(45)
        % xticks(0:60:unique_time(end))
        title('dT/dt [°C/s]')

        % ➕ New Grouped Subplot by Material Type        
        % ➕ New Grouped Subplot by Material Type
        figure(3)
        subplot(2,2,color_idx)
        hold on
        plot(time, temperature, '.', 'Color', colors(color_idx,:))
        title({'dT/dt'; ['Material: ', notes{1}]}, 'Interpreter', 'none')
        xlabel('Time [s]')
        ylabel('dT/dt [°C/s]')
        grid on
        figure(4)
        subplot(2,2,color_idx)
        hold on
        plot(time, dtemperature, '.', 'Color', colors(color_idx,:))
        title({'dT/dt'; ['Material: ', notes{1}]}, 'Interpreter', 'none')
        xlabel('Time [s]')
        ylabel('dT/dt [°C/s]')
        grid on
    end



end


return;
%% NN =========================================================

fprintf("Preparing Dataset...\n");


trainData = [];trainLbl = [];
testData = [];testLbl = [];
classNames = [];
for file_idx = 1:num_of_file

    % Load one file
    filename = fileNames{file_idx};
    fprintf("Processing " + filename + "\n");
    t = T{filename};

    % Get unique experiments
    exps = unique(t.Experiment);
    % Get Label names
    classNames = [classNames; string(unique(t.notes))];

    % First 10 experiments for training
    trainIDs = [1,2,3,4,5,6,7,9,10,11];
    isTrain = ismember(t.Experiment, trainIDs);

    % %remove outliers
    % for i = 1:15
    %     idx = ismember(t.Experiment, i);
    %     t(idx, :).temperature_C_ = filloutliers(t(idx, :).temperature_C_, 'linear','threshold', 50);
    %     % t(idx, :).temperature_C_ = smoothdata(t(idx, :).temperature_C_, 'movmedian', 10);
    %     % t(idx, :).temperature_C_ = rmoutliers(t(idx, :).temperature_C_);
    % 
    % end

    % resample
    seq = 1:10; % resample each 10 sample
    repeatedSeq = repmat(seq, 1, ceil(height(t)/max(seq)));
    t.resample = repeatedSeq(1:height(t))';

    % Split
    trainData = [trainData; t(isTrain, :)];
    testData  = [testData; t(~isTrain, :)];

    % trainLbl = [trainLbl; file_idx];
    % testLbl = [testLbl; file_idx];
    % trainLbl = [trainLbl; repmat(file_idx,10,1)];
    % testLbl = [testLbl; repmat(file_idx,5,1)];
    trainLbl = [trainLbl; repmat(file_idx,height(t(isTrain, :)),1)];
    testLbl = [testLbl; repmat(file_idx,height(t(~isTrain, :)),1)];
end

% Group by experiment
[testGroups, testID] = findgroups(testData.resample, testData.Experiment, testData.notes);
[trainGroups, trainID] = findgroups(trainData.resample, trainData.Experiment, trainData.notes);

% Split into cell arrays of sequences
XTrain = splitapply(@(x){x}, trainData.temperature_C_, trainGroups);
TTrain = categorical(cell2mat(splitapply(@(x){x(1)}, trainLbl, trainGroups)));
% TTrain = categorical(trainLbl);

XTest = splitapply(@(x){x}, testData.temperature_C_, testGroups);
TTest = categorical(cell2mat(splitapply(@(x){x(1)}, testLbl, testGroups)));
% TTest = categorical(testLbl);

% trainData = [];
% trainLbl = [];
% 
% for file_idx = 1:length(fileNames)
%     filename = fileNames{file_idx};
%     t = T{filename};  
% 
%     exps = unique(t.Experiment);
%     trainIDs = exps(1:min(10, end));
%     isTrain = ismember(t.Experiment, trainIDs);
% 
%     curTrain = t(isTrain, :);
% 
%     trainData = [trainData; curTrain];
% 
%     % Append labels per experiment
%     for e = 1:length(trainIDs)
%         trainLbl = [trainLbl; file_idx];  % or assign label according to experiment
%     end
% end
% 
% trainGroups = findgroups(trainData.Experiment);
% XTrain = splitapply(@(x){x}, trainData.temperature_C_, trainGroups);
% TTrain = trainLbl%categorical(trainLbl);


figure(3)
hold on;
for i = 1:length(XTrain)
    plot(XTrain{i},'Color', colors(TTrain(i),:));
end
hold off;
xlabel('Time Step');
ylabel('Temperature (C)');
title('Training Sequences: Temperature over Time');
figure(5)
hold on;
for i = 1:length(XTest)
    plot(XTest{i},'Color', colors(TTest(i),:));
end
hold off;
xlabel('Time Step');
ylabel('Temperature (C)');
title('Test Sequences: Temperature over Time');


%% 
numFeatures = 1;
numHiddenUnits = 200;
numClasses = 5;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,OutputMode="last")
    fullyConnectedLayer(numClasses)
    softmaxLayer];

% 
% numHiddenUnits1 = 125;
% numHiddenUnits2 = 100;
% layers = [ ...
%     sequenceInputLayer(numFeatures)
%     lstmLayer(numHiddenUnits1,OutputMode="sequence")
%     dropoutLayer(0.2)
%     lstmLayer(numHiddenUnits2,OutputMode="last")
%     dropoutLayer(0.2)
%     fullyConnectedLayer(numClasses)
%     softmaxLayer];
% 
% options = trainingOptions("adam", ...
%     MaxEpochs=1000, ...
%     InitialLearnRate=1e-5,...
%     GradientThreshold=1, ...
%     Shuffle="never", ...
%     Plots="training-progress", ...
%     Metrics="accuracy", ...
%     Verbose=false);

fprintf("Training Network...\n");


% mu = mean([XTrain{:}],1);
% sigma = std([XTrain{:}],0,1);
% XTrain = cellfun(@(X) (X-mu)./sigma,XTrain,UniformOutput=false);

net = trainnet(XTrain,TTrain,layers,"crossentropy",options);
%%
scores = minibatchpredict(net,XTest);
YTest = scores2label(scores,classNames);
figure(4)
confusionchart(categorical(classNames(TTest)),YTest)
figure(5)
hold on;
for i = 1:length(XTest)
    if YTest(i)==categorical(classNames(TTest(i)))
        plot(XTest{i}, 'Color', colors(YTest(i),:));
    else
        plot(XTest{i},'x','Color', 'black');
    end
end
hold off;
xlabel('Time Step');
ylabel('Temperature (C)');
title('Test Sequences: Temperature over Time');
fprintf("Classification Score: %.0f%%\n",sum(YTest==categorical(classNames(TTest)))/length(TTest)*100);