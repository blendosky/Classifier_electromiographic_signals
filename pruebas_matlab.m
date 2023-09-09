close;
close all;
clear;
 
data = importdata("datashets\1.txt");

data = data.data;

data_norm = normalize(data(:,2:9), "range",[-1,1]);
data_norm = [data_norm data(:,10)];

figure(1)
subplot(2,1,1)
plot(data_norm(:,1))
subplot(2,1,2)
plot(data_norm(:,9))
grid on;

t = data(:,10)';
x = data_norm(:,1:8)';

L = length(t);

net = feedforwardnet([50 25]);

net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';

% %Parámetros de entrenamiento (criterios de parada)

net.trainParam.max_fail = 50;                       %6
net.trainParam.epochs = 100;                       %1000
%net.trainParam.goal =  1e-20;                      %0	
%net.trainParam.min_grad = 1e-10;                   %1e-7

net.divideFcn = 'dividerand';

[trainInd,valInd,testInd] = dividerand(L,0.7*L,0.2*L,0.1*L);

[net,info,y, e] = train(net,x,t);
view(net)

y_class = sim(net,x);
y_class = round(y_class);

figure(2)
plot(y_class)
hold on;
plot(t)

