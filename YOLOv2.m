doTraining= false;
%% Read file and create table
% Set input folder
input_folder = '/Users/raghad/Desktop/200images!/OneDrive_6_5-10-2020/RaghadLabels_200';
%input_folder = '/Users/raghad/Desktop/gp matlab/CheckedDataset5';
 
% Read all *.jpg files from input folder
img_files = dir(fullfile(input_folder, '*.JPEG'));
% Get full path names for each text file
img_paths = fullfile({img_files.folder}, {img_files.name});
imageFilename = cellstr(img_paths')
imageFilename = sortrows(imageFilename) 
 
%read all BB index txt files
BB_files = dir(fullfile(input_folder, '*.txt'));
% Get full path names for each text file
BB_paths = fullfile({BB_files.folder}, {BB_files.name});
BB_paths = cellstr(BB_paths')
BB_paths = sortrows(BB_paths)
 


for i=1:length(BB_paths)
 
    fileID = fopen(BB_paths{i},'r');
 
    BB_txt = textscan(fileID, '%s');
 
    %remove the txt 'Handgun' from the cell elements
 
    %BB_txt{1,1} = []
 
    BBindx = str2double(BB_txt{1,1});
 
    %BBindx(1,:) = []
 
    n=(BBindx(1));  %% Number of bounding boxes in image
 
    %convert double values to int
 
   
 
    if (n ~= 1 )
 
        counter=2;
 
        temp=zeros(n,4);
 
    for j=1:n
 
        
 
    x1=(BBindx(counter))+1;
 
    y1=(BBindx(counter+1))+1;
 
    x2=(BBindx(counter+2))+1;
 
    y2=(BBindx(counter+3))+1;
 
    counter = counter+4;
 
    
 
    w=x2-x1+1;
 
    h=y2-y1+1;
 
    
 x1=round(x1);
    y1=round(y1);
    w=round(w);
    h=round(h);
    temp(j,:)=[x1, y1, w, h];
 
    
 
    %store values in row vector
 
    end
 
    
 
    counter=2;
 
    guns(i,:) = {temp};
 
    
 
    else
 
        
 
    x1=(BBindx(2))+1;
 
    y1=(BBindx(3))+1;
 
    x2=(BBindx(4))+1;
 
    y2=(BBindx(5))+1;
 
    
 
    w=x2-x1+1;
 
    h=y2-y1+1;
 
     x1=round(x1);
    y1=round(y1);
    w=round(w);
    h=round(h);
        guns(i,:) = {[x1 y1 w h;]};
 
    end
 
end
 
%create table of image paths and gun BB
gunDataset = table(imageFilename, guns);
summary(gunDataset)
 
 
 
%% Load Dataset

 
%choose dataset to pass to the network (guns/vichels)
dataset = gunDataset %% gunDataset %%vehicleDataset 
lable = 'guns'%%'guns' %% 'vehicle' 
 
 
 
rng(0);

 
%idx = floor(0.6 * length(shuffledIndices) );
 
idx = 1785;

trainingDataTbl = dataset(1:idx,:);
testDataTbl = dataset(idx+1:end,:);

imdsTrain = imageDatastore(trainingDataTbl{:,1});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,2));

imdsTest = imageDatastore(testDataTbl{:,1});
bldsTest = boxLabelDatastore(testDataTbl(:,2));
 
trainingData = combine(imdsTrain,bldsTrain);
testData = combine(imdsTest,bldsTest);
 
%data = read(trainingData);
%I = data{1};
%bbox = data{2};
%annotatedImage = insertShape(I,'Rectangle',bbox);
%annotatedImage = imresize(annotatedImage,2);
%figure
%imshow(annotatedImage)
 
inputSize = [224 224 3];
 
 
numClasses = width(dataset)-1;
 
 
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
 
numAnchors = 3;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)
 
featureExtractionNetwork = resnet50;
 
featureLayer = 'activation_40_relu';
 
lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);
 
augmentedTrainingData = transform(trainingData,@augmentData);
 
% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,'BorderSize',10)
 
preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));
 
data = read(preprocessedTrainingData);
 
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'Rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)
 
options = trainingOptions('sgdm', ...
        'MiniBatchSize', 16, ....
        'InitialLearnRate',1e-3, ...
        'MaxEpochs',20,...
        'CheckpointPath', tempdir, ...
        'Shuffle','never');
    
 if doTraining       
     tic;
    % Train the YOLO v2 detector.
    [detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
     %%%%%saving
    finalmodel = detector ;
    save finalmodel
    disp("MODEL SAVED");
     toc;
else
    % Load pretrained detector for the example.
    pretrained = load('MediumDataset_YOLOv2.mat');
    detector = pretrained.detector;
 end
 
 I = imread(testDataTbl.imageFilename{1});
I = imresize(I,inputSize(1:2));
[bboxes,scores] = detect(detector,I);
 
I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)
preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));
 
tic;
detectionResults = detect(detector, preprocessedTestData);
disp("Testing time")
toc;
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);
 
avg_recall=mean(recall)
avg_precision=mean(precision)
 
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
 
 
[am, fppi, missRate] = evaluateDetectionMissRate(detectionResults, preprocessedTestData);
figure;
loglog(fppi, missRate);
grid on
title(sprintf('Log Average Miss Rate = %.1f', am))
 
 
 
 
function B = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.
B = cell(size(A));
 
I = A{1};
sz = size(I);
if numel(sz)==3 && sz(3) == 3
    I = jitterColorHSV(I,...
        'Contrast',0.2,...
        'Hue',0,...
        'Saturation',0.1,...
        'Brightness',0.2);
end
 
% Randomly flip and scale image.
tform = randomAffine2d('XReflection',true,'Scale',[1 1.1]);
rout = affineOutputView(sz,tform,'BoundsStyle','CenterOutput');
B{1} = imwarp(I,tform,'OutputView',rout);
 
% Apply same transform to boxes.
[B{2},indices] = bboxwarp(A{2},tform,rout,'OverlapThreshold',0.25);
B{3} = A{3}(indices);
 
% Return original data only when all boxes are removed by warping.
if isempty(indices)
    B = A;
end
end
 
function data = preprocessData(data,targetSize)
% Resize image and bounding boxes to the targetSize.
scale = targetSize(1:2)./size(data{1},[1 2]);
data{1} = imresize(data{1},targetSize(1:2));
data{2} = bboxresize(data{2},scale);
end
