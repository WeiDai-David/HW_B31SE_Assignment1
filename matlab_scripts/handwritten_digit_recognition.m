% https://uk.mathworks.com/help/deeplearning/examples/create-simple-deep-learning-network-for-classification.html
rng(0);   % Fixed random number seed, making split and initialization more stable
% Load Image Data
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Display some of the images in the datastore	
figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

% Calculate the number of images in each category
labelCount = countEachLabel(imds)

% Each image is 28-by-28-by-1 pixels.
img = readimage(imds,1);
size(img)

% Specify Training and Validation Sets
% Divide the data into training and validation data sets, 
% so that each category in the training set contains 750 images, 
% and the validation set contains the remaining images from each label. 
% splitEachLabel splits the datastore digitData into two new datastores, 
% trainDigitData and valDigitData.
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

% Define the convolutional neural network architecture
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

% Specify Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train Network Using Training Data
net = trainNetwork(imdsTrain,layers,options);

% Classify Validation Images and Compute Accuracy
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
